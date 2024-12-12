library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)
library(embed)
library(themis)
library(lme4)
library(stacks)

cl <- makePSOCKcluster(25)
registerDoParallel(cl)

# Load in data and sample submission from Kaggle Competition
data <- vroom("./data.csv")
sample_sub <- vroom("./sample_submission.csv")

# Separate data into train and test datasets
test_data <- filter(data, is.na(shot_made_flag))
# test_data <- select(test_data, c(-shot_made_flag))

train_data <- filter(data, !is.na(shot_made_flag))

# Change shot_made_flag to factor
train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)

# Add home vs. away columns to test and train
train_data$home_court <- ifelse(str_detect(train_data$matchup, "vs."),
                                "yes",
                                "no")

test_data$home_court <- ifelse(str_detect(test_data$matchup, "vs."),
                                "yes",
                                "no")

# Make season numeric to test and train
train_data$season <- as.numeric(substr(train_data$season,6,7)) + 3
train_data$season <- ifelse(train_data$season > 90,
                            train_data$season - 100,
                            train_data$season)

test_data$season <- as.numeric(substr(test_data$season,6,7)) + 3
test_data$season <- ifelse(test_data$season > 90,
                           test_data$season - 100,
                           test_data$season)

# Create Recipe & Perform Feature Engineering
kb24_recipe <- recipe(shot_made_flag ~ ., data = train_data) %>% 
  step_mutate(time_remaining = (60 * minutes_remaining) + seconds_remaining) %>%
  step_mutate(angle = ifelse(loc_x == 0, pi / 2, atan(loc_y / loc_x))) %>% 
  step_mutate(distance = sqrt((loc_x/10)^2+(loc_y/10)^2)) %>% 
  step_mutate(game_num = as.numeric(game_date)) %>% 
  step_rm(game_date, team_name, matchup, team_id, minutes_remaining, 
          lon, lat, action_type, shot_id, game_id,
          seconds_remaining, loc_x, loc_y, matchup, shot_zone_area,
          shot_zone_range, shot_zone_basic) %>%
  step_mutate(playoffs = as.factor(playoffs),
              season = as.factor(season),
              home_court = as.factor(home_court),
              period = as.factor(period),
              shot_type = as.factor(shot_type), 
              opponent = as.factor(opponent),
              season = as.factor(season)) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors())
prepped_kb24_recipe <- prep(kb24_recipe)
bake(prepped_kb24_recipe, new_data = train_data)

## Stacking
# Split data for CV
folds <- vfold_cv(train_data, v = 3, repeats = 1)

# Create a control grid
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Penalized regression model
plog_reg_model <- logistic_reg(penalty = tune(),
                               mixture = tune()) %>% 
  set_engine("glmnet")

# Set Workflow
plog_reg_wf <- workflow() %>% 
  add_recipe(kb24_recipe) %>% 
  add_model(plog_reg_model)

# Grid of tuning parameter values
plog_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5)

# Run CV
plog_reg_models <- plog_reg_wf %>% 
  tune_grid(resamples = folds,
            grid = plog_tuning_grid,
            metrics = metric_set(roc_auc),
            control = untunedModel)

# Random Forest Models
srf_model <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Create workflow with model & recipe
srf_wf <- workflow() %>% 
  add_recipe(kb24_recipe) %>% 
  add_model(srf_model)


final_par <- extract_parameter_set_dials(srf_model) %>% 
  finalize(train_data)

# Set tuning grid
tuning_grid <- grid_regular(final_par,
                            levels = 5)

# Run Cv
srf_models <- srf_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc),
            control = untunedModel)

# Specify which models to include
my_stack <- stacks() %>% 
  add_candidates(plog_reg_models) %>% 
  add_candidates(srf_models)

# Fit the stacked model
stack_mod <- my_stack %>% 
  blend_predictions() %>% 
  fit_members()

# Use stacked data to get prediction
stack_preds <- stack_mod %>% 
  predict(new_data = test_data,
          type = "prob")

# Format predictions for kaggle submission
kaggle_submission <- stack_preds %>% 
  bind_cols(., test_data) %>% 
  select(shot_id, .pred_1) %>% 
  rename(shot_made_flag = .pred_1)

# Export kaggle submission

vroom_write(x = kaggle_submission,
            file = "./stack_preds.csv", 
            delim = ",")

stopCluster(cl)