library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(ggplot2)
library(GGally)
library(discrim)
library(embed)
library(themis)
library(stringr)
library(bonsai)
library(lightgbm)

# Load in data and sample submission from Kaggle Competition
data <- vroom("./data.csv")
sample_sub <- vroom("./sample_submission.csv")

# Separate data into train and test datasets
test_data <- filter(data, is.na(shot_made_flag))
# test_data <- select(test_data, c(-shot_made_flag))
                
train_data <- filter(data, !is.na(shot_made_flag))

# Change shot_made_flag to factor
train_data$shot_made_flag <- as.factor(train_data$shot_made_flag)

## EDA
glimpse(train_data)
skim(train_data)
plot_intro(train_data)

plot_correlation(select(train_data, c(playoffs, shot_type, shot_distance, shot_zone_area, shot_zone_basic, shot_zone_range)))# game_event_id may have zero variance

plot_missing(train_data) # No missing data
plot_histogram(train_data)


# Bar plot of shot_made_flag to check for disproportionate amounts of response
plot1 <- ggplot(data = train_data,
                mapping = aes(x = shot_made_flag)) +
  geom_bar()
plot1 # response does not appear to be disproporionately distributed

# Bar plot of team_name
plot2 <- ggplot(data = train_data,
                mapping = aes(x = team_name)) +
  geom_bar()
plot2 # Only one value for team_name, so remove team_name field

# Bar plot of team_id
plot3 <- ggplot(data = train_data,
                mapping = aes(x = team_id)) +
  geom_bar()
plot3 # Only one value for team_id, so remove team_id field

# Bar plot of shot_zone_basic
plot4 <- ggplot(data = train_data,
                mapping = aes(x = shot_zone_basic)) +
  geom_bar()
plot4 

# Bar plot of shot_zone_area
plot5 <- ggplot(data = train_data,
                mapping = aes(x = shot_zone_area)) +
  geom_bar()
plot5

# Bar plot of shot_zone_range
plot6 <- ggplot(data = train_data,
                mapping = aes(x = shot_zone_range)) +
  geom_bar()
plot6

# Bar plot of playoffs
plot7 <- ggplot(data = train_data,
                mapping = aes(x = playoffs)) +
  geom_bar()
plot7

# Bar plot of action_type
plot8 <- ggplot(data = train_data,
                mapping = aes(x = action_type)) +
  geom_bar()
plot8

# Bar plot of combined_shot_type
plot9 <- ggplot(data = train_data,
                mapping = aes(x = combined_shot_type)) +
  geom_bar()
plot9

# Bar plot of game_event_id
plot10 <- ggplot(data = train_data,
                mapping = aes(x = game_event_id)) +
  geom_histogram()
plot10

# Bar plot of game_id
plot11 <- ggplot(data = train_data,
                 mapping = aes(x = combined_shot_type)) +
  geom_bar()
plot11



# Create Recipe & Perform Feature Engineering
kb24_recipe <- recipe(shot_made_flag ~ ., data = train_data) %>% 
  step_date(game_date, features = c("year","decimal","month")) %>%
  step_rm(game_date, team_name, matchup, team_id, lon, lat) %>%
  step_mutate(game_date_year = as.factor(game_date_year)) %>% 
  step_mutate(game_date_month = as.factor(game_date_month)) %>%
  step_mutate(action_type = as.factor(action_type)) %>% 
  step_mutate(combined_shot_type = as.factor(combined_shot_type)) %>% 
  step_mutate(shot_zone_basic = as.factor(shot_zone_basic)) %>% 
  step_mutate(shot_zone_area = as.factor(shot_zone_area)) %>% 
  step_mutate(shot_zone_range = as.factor(shot_zone_range)) %>% 
  step_mutate(playoffs = as.factor(playoffs)) %>% 
  step_mutate(shot_type = as.factor(shot_type)) %>% 
  step_mutate(opponent = as.factor(opponent)) %>% 
  step_mutate(season = as.factor(season)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())
prepped_kb24_recipe <- prep(kb24_recipe)
baked_data <- bake(prepped_kb24_recipe, new_data = train_data)

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
tuning_grid <- grid_regular(Laplace(range = c(1,5)),
                            smoothness(range = c(0.5,1.5)),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "accuracy")
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

# Random Forest Model
rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 10) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

# Set workflow
rf_wf <- workflow() %>% 
  add_recipe(kb24_recipe) %>% 
  add_model(rf_model)

# Finalize mtry() parameter
final_par <- extract_parameter_set_dials(rf_model) %>% 
  finalize(train_data)

# Set tuning grid
tuning_grid <- grid_regular(final_par,
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "accuracy")
best_tune

# Finalize workflow
rf_final_wf <- rf_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
rf_preds <- predict(rf_final_wf,
                    new_data = test_data,
                    type = "prob")


# Boosted Trees Model
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("lightgbm")

# Set workflow
boost_wf <- workflow() %>% 
  add_recipe(kb24_recipe) %>% 
  add_model(boost_model)

# Set tuning grid
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "accuracy")
best_tune

# Finalize workflow
boost_final_wf <- boost_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
boost_preds <- predict(boost_final_wf,
                       new_data = test_data,
                       type = "class")

# Facebook Prophet Model

cv_split <- time_series_split(train_data,
                              assess = "1 month",
                              cumulative = TRUE)

# Define FBProphet model
FBP_model <- prophet_reg() %>% 
  set_engine(engine = "prophet") %>% 
  fit(sales ~ date, data = training(cv_split))

# Calibrate (tune) the models
cv_results <- modeltime_calibrate(FBP_model,
                                  new_data = testing(cv_split))

# Visualize results
cv_plot <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = training(cv_split)) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

# Refit to whole dataset
fullfit <- cv_results %>% 
  modeltime_refit(data = train_data)

# Predict for all the observations in storeItemTest1
fullfit_plot <- fullfit %>% 
  modeltime_forecast(new_data = test_data,
                     actual_data = train_data) %>% 
  plot_modeltime_forecast(.interactive = FALSE)



