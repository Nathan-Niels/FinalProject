library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)
library(embed)
library(themis)

# Load in data and sample submission from Kaggle Competition
data <- vroom("./data.csv")

# Add home court advantage predictor
data$matchup <- ifelse(str_detect(data$matchup, "vs."),
                       "yes",
                       "no")
# Make season numeric
data$season <- as.numeric(substr(data$season,6,7))

# Separate data into train and test datasets
train_data <- data[is.na(data$shot_made_flag) == FALSE,]

test_data <- data[is.na(data$shot_made_flag) == TRUE,]

# Change shot_made_flag to factor
train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)

# Create Recipe & Perform Feature Engineering
kb24_recipe <- recipe(shot_made_flag ~ ., data = train_data) %>% 
  step_mutate(time_remaining = (60 * minutes_remaining) + seconds_remaining) %>%
  step_mutate(angle = case_when(loc_x == 0 ~ pi / 2, TRUE ~ atan(loc_y / loc_x))) %>% 
  step_mutate(shot_distance = sqrt((loc_x/10)^2+(loc_y/10)^2)) %>% 
  step_mutate(game_num = as.numeric(game_date)) %>% 
  step_mutate(playoffs = as.factor(playoffs),
              game_id = as.factor(game_id),
              shot_zone_area = as.factor(shot_zone_area),
              shot_zone_basic = as.factor(shot_zone_basic),
              matchup = as.factor(matchup),
              season = as.factor(season),
              shot_id = as.factor(shot_id),
              period = as.factor(period),
              shot_type = as.factor(shot_type), 
              opponent = as.factor(opponent)) %>%
  step_rm(game_date, team_name, team_id, minutes_remaining, 
          lon, lat, shot_id, game_id, game_event_id,
          seconds_remaining, loc_x, loc_y, shot_zone_area,
          shot_zone_basic, shot_zone_range) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors())
prepped_kb24_recipe <- prep(kb24_recipe)
bake(prepped_kb24_recipe, new_data = train_data)

# NB Model
nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

# Set workflow
nb_wf <- workflow() %>% 
  add_recipe(kb24_recipe) %>% 
  add_model(nb_model)

# Set tuning grid
tuning_grid <- grid_regular(Laplace(c(1,5)),
                            smoothness(c(0.5,1.5)),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
nb_final_wf <- nb_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
nb_preds <- predict(nb_final_wf,
                    new_data = test_data,
                    type = "prob")

# Format predictions for kaggle submission
kaggle_submission <- nb_preds %>% 
  bind_cols(., test_data) %>% 
  select(shot_id, .pred_1) %>% 
  rename(shot_made_flag = .pred_1)

# Export kaggle submission

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/FinalProject/nb_preds.csv", 
            delim = ",")
