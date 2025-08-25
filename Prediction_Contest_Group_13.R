library(dplyr)
library(stringr)
library(geodist)
library(fastDummies)
library(rpart.plot)
library(caret)
library(randomForest)
library(xgboost)

#####################################
#PREPROCESSING AND CLEANING THE DATA
#####################################

austinhouses <- read.csv("austinhouses.csv", stringsAsFactors = FALSE)
austinhouses_holdout <- read.csv("austinhouses_holdout.csv", stringsAsFactors = FALSE)

#clean the data
housingDF <- austinhouses
#Create downtown coordinates
downtownLat <- 30.2747
downtownLon <- -97.7404
#Create new predictors from description
housingDF <- housingDF %>%
  mutate(
    hasPool        = str_detect(tolower(description), "\\bpool\\b"),
    isGated        = str_detect(tolower(description), "\\bgated\\b"),
    isEstate       = str_detect(tolower(description), "\\bestate\\b"),
    hasFireplace   = str_detect(tolower(description), "\\bfireplace\\b"),
    isInCuldesac     = str_detect(tolower(description), "\\bcul[-\\s]?de[-\\s]?sac\\b"),
    mentionsNeighborhood  = str_detect(tolower(description), "\\bneighbou?rhood\\b"),
    nearLake       = str_detect(tolower(description), "\\blake\\b"),
    nearRiver      = str_detect(tolower(description), "\\briver\\b"),
    nearHighway    = str_detect(tolower(description), "\\bhighway\\b"),
    nearPlayground = str_detect(tolower(description), "\\bplayground\\b"),
    graniteCounters = str_detect(tolower(description), "\\bgranite\\b"),
    mentionsPaint = str_detect(tolower(description), "\\bpaint(ed)?\\b"),
    remodeledOrRenovated = str_detect(tolower(description), "\\b(renovat(ed|ion)|remodel(ed)?)\\b"),
    #Calculate the miles away from downtown
    milesFromDowntown = geodist(cbind(longitude, latitude),cbind(downtownLon, downtownLat), measure = "haversine") / 1609.34,
    #Get the average teachers per school
    avgTeachersPerSchool = avgSchoolSize / MedianStudentsPerTeacher
  )
housingDF <- dummy_cols(housingDF ,select_columns = "zipcode",remove_first_dummy = TRUE,remove_selected_columns = TRUE)
#Drop the irrelevant columns

#Add some relevant interaction terms
housingDF <- housingDF %>% select(-streetAddress, -description, -homeType, -latest_saledate)
housingDF$livingArea_year <- housingDF$livingAreaSqFt * housingDF$yearBuilt
housingDF$lot_pool <- housingDF$lotSizeSqFt * housingDF$hasPool
housingDF$distance_school <- housingDF$milesFromDowntown * housingDF$avgSchoolRating
housingDF$garage_combo <- housingDF$garageSpaces * housingDF$hasGarage
housingDF$area_bedrooms <- housingDF$livingAreaSqFt * housingDF$numOfBedrooms
housingDF$spa_pool <- housingDF$hasSpa * housingDF$hasPool
housingDF$granite_reno <- housingDF$graniteCounters * housingDF$remodeledOrRenovated
housingDF$school_quality_access <- housingDF$avgSchoolRating * housingDF$avgSchoolDistance
housingDF$tax_age <- housingDF$propertyTaxRate * housingDF$yearBuilt
housingDF$bed_bath_ratio <- housingDF$numOfBathrooms * housingDF$numOfBedrooms

#training and testing set
set.seed(7)
n <- nrow(housingDF)
test_indices <- sample(n, size = floor(0.2 * n))
test <- housingDF[test_indices, ]
train <- housingDF[-test_indices, ]

#####################################
#STEPWISE SELECTION (MAIN)
#####################################

full = lm(latestPrice~., data=train)

# Start from the full model, and consider adding/deleting
# any single term at each step
stepwise <- stats::step(full, direction="both")

formula_bag <- log(latestPrice) ~ latitude + garageSpaces + hasAssociation + hasGarage + 
  yearBuilt + latest_saleyear + numOfPhotos + numOfAccessibilityFeatures + 
  numOfPatioAndPorchFeatures + numOfWaterfrontFeatures + numOfWindowFeatures + 
  lotSizeSqFt + avgSchoolRating + avgSchoolSize + MedianStudentsPerTeacher + 
  numOfBathrooms + numOfBedrooms + numOfStories + hasPool + 
  isEstate + nearLake + nearRiver + nearPlayground + graniteCounters + 
  mentionsPaint + remodeledOrRenovated + milesFromDowntown + 
  avgTeachersPerSchool + zipcode_78702 + zipcode_78703 + zipcode_78704 + 
  zipcode_78705 + zipcode_78721 + zipcode_78722 + zipcode_78723 + 
  zipcode_78724 + zipcode_78725 + zipcode_78726 + zipcode_78727 + 
  zipcode_78728 + zipcode_78729 + zipcode_78731 + zipcode_78732 + 
  zipcode_78734 + zipcode_78736 + zipcode_78741 + zipcode_78744 + 
  zipcode_78745 + zipcode_78746 + zipcode_78747 + zipcode_78748 + 
  zipcode_78749 + zipcode_78752 + zipcode_78753 + zipcode_78754 + 
  zipcode_78756 + zipcode_78758 + zipcode_78759 + livingArea_year + 
  lot_pool + distance_school + area_bedrooms + school_quality_access + 
  bed_bath_ratio

num_predictors <- length(all.vars(update(formula_bag, . ~ .))) - 1

#####################################
#DECISION TREES
#####################################

#initial tree
tree_austinhouses <- rpart(formula_bag, data=train, control = rpart.control(cp = 0.0001))
rpart.plot(tree_austinhouses, type = 3, extra = 1, fallen.leaves = TRUE)

pred_values <- exp(predict(tree_austinhouses, test))
actual_values <- test$latestPrice
test_rmse <- sqrt(mean((pred_values - actual_values)^2))
test_rmse
#196.86


#pruned tree
min_xerror_row <- which.min(tree_austinhouses$cptable[, "xerror"])
optimal_cp_min <- tree_austinhouses$cptable[min_xerror_row, "CP"]
tree_ah_pruned <- rpart::prune(tree_austinhouses, cp = optimal_cp_min)

pred_pruned_values <- exp(predict(tree_ah_pruned, test))
actual_pruned_values <- test$latestPrice
test_rmse_pruned <- sqrt(mean((pred_pruned_values - actual_pruned_values)^2))
test_rmse_pruned
#194.56

#####################################
# BAGGING MODEL
#####################################

#Bagging
library(randomForest)
set.seed(7)
bag_austinhouses <- randomForest(formula_bag, data=train, mtry = num_predictors)
pred_bag <- exp(predict(bag_austinhouses, test))
rmse_bag <- sqrt(mean((pred_bag - test$latestPrice)^2))
rmse_bag
#167.2

#####################################
#RANDOM FOREST MODEL
#####################################

set.seed(7)
rf_mtry_1 <- randomForest(formula_bag, data=train, ntree = 500, mtry = floor(sqrt(num_predictors)))
pred_mtry_1 <- exp(predict(rf_mtry_1, test))
rmse_mtry_1 <- sqrt(mean((pred_mtry_1 - test$latestPrice)^2))
rmse_mtry_1
#166.61

#####################################
#XGBOOST MODEL
#####################################

predictors <- c(
  "latitude", "garageSpaces", "hasAssociation", "hasGarage",
  "yearBuilt", "latest_saleyear", "numOfPhotos", "numOfAccessibilityFeatures",
  "numOfPatioAndPorchFeatures", "numOfWaterfrontFeatures", "numOfWindowFeatures",
  "lotSizeSqFt", "avgSchoolRating", "avgSchoolSize", "MedianStudentsPerTeacher",
  "numOfBathrooms", "numOfBedrooms", "numOfStories", "hasPool",
  "isEstate", "nearLake", "nearRiver", "nearPlayground", "graniteCounters",
  "mentionsPaint", "remodeledOrRenovated", "milesFromDowntown", "avgTeachersPerSchool",
  "zipcode_78702", "zipcode_78703", "zipcode_78704", "zipcode_78705",
  "zipcode_78721", "zipcode_78722", "zipcode_78723", "zipcode_78724",
  "zipcode_78725", "zipcode_78726", "zipcode_78727", "zipcode_78728",
  "zipcode_78729", "zipcode_78731", "zipcode_78732", "zipcode_78734",
  "zipcode_78736", "zipcode_78741", "zipcode_78744", "zipcode_78745",
  "zipcode_78746", "zipcode_78747", "zipcode_78748", "zipcode_78749",
  "zipcode_78752", "zipcode_78753", "zipcode_78754", "zipcode_78756",
  "zipcode_78758", "zipcode_78759", "livingArea_year", "lot_pool",
  "distance_school", "area_bedrooms", "school_quality_access", "bed_bath_ratio"
)

library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(train[, predictors]), label = log(train$latestPrice))   
dtest <- xgb.DMatrix(data = as.matrix(test[, predictors]))

set.seed(7)
xgb_model <- xgboost(
  data = dtrain,
  objective = "reg:squarederror", 
  nrounds = 100,                  
  eta = 0.1,                      
  max_depth = 6,                  
  verbose = 0                    
)

pred_log_prices <- predict(xgb_model, newdata = dtest)
pred_prices <- exp(pred_log_prices) 
actual_prices <- test$latestPrice
rmse_xgb <- sqrt(mean((pred_prices - actual_prices)^2))
cat("XGBoost RMSE:", round(rmse_xgb, 2), "\n")
#165.32

# Set up tuning grid
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1),
  gamma = 0,                
  colsample_bytree = 1,   
  min_child_weight = 1,
  subsample = 1
)

train_control <- trainControl(method = "cv", number = 5)

# Train using caret
set.seed(7)
xgb_tuned <- train(
  x = as.matrix(train[, predictors]),
  y = log(train$latestPrice),
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  verbose = FALSE
)

# Best tuning parameters
xgb_tuned$bestTune

#Fitting with the best tuning parameters
best_params <- xgb.DMatrix(data = as.matrix(train[, predictors]), label = log(train$latestPrice))

xgb_final <- xgboost(
  data = best_params,
  nrounds = 200,
  max_depth = 6,
  eta = 0.1,
  objective = "reg:squarederror",
  verbose = 0
)

dtest <- xgb.DMatrix(data = as.matrix(test[, predictors]))
xgb_preds <- predict(xgb_final, newdata = dtest)
xgb_rmse <- sqrt(mean((exp(xgb_preds) - test$latestPrice)^2))
cat("XGBoost RMSE on Test Set:", round(xgb_rmse, 2), "\n")
#164.02


#####################################
#PREDICTING ON HOLD OUT DATA
#####################################

#clean the data
holdoutDF <- austinhouses_holdout
#Create downtown coordinates
downtownLat <- 30.2747
downtownLon <- -97.7404
#Create new predictors from description
holdoutDF <- holdoutDF %>%
  mutate(
    hasPool        = str_detect(tolower(description), "\\bpool\\b"),
    isGated        = str_detect(tolower(description), "\\bgated\\b"),
    isEstate       = str_detect(tolower(description), "\\bestate\\b"),
    hasFireplace   = str_detect(tolower(description), "\\bfireplace\\b"),
    isInCuldesac     = str_detect(tolower(description), "\\bcul[-\\s]?de[-\\s]?sac\\b"),
    mentionsNeighborhood  = str_detect(tolower(description), "\\bneighbou?rhood\\b"),
    nearLake       = str_detect(tolower(description), "\\blake\\b"),
    nearRiver      = str_detect(tolower(description), "\\briver\\b"),
    nearHighway    = str_detect(tolower(description), "\\bhighway\\b"),
    nearPlayground = str_detect(tolower(description), "\\bplayground\\b"),
    graniteCounters = str_detect(tolower(description), "\\bgranite\\b"),
    mentionsPaint = str_detect(tolower(description), "\\bpaint(ed)?\\b"),
    remodeledOrRenovated = str_detect(tolower(description), "\\b(renovat(ed|ion)|remodel(ed)?)\\b"),
    #Calculate the miles away from downtown
    milesFromDowntown = geodist(cbind(longitude, latitude),cbind(downtownLon, downtownLat), measure = "haversine") / 1609.34,
    #Get the average teachers per school
    avgTeachersPerSchool = avgSchoolSize / MedianStudentsPerTeacher
  )
holdoutDF <- dummy_cols(holdoutDF ,select_columns = "zipcode",remove_first_dummy = TRUE,remove_selected_columns = TRUE)
#Drop the irrelevant columns
holdoutDF <- holdoutDF %>% select(-streetAddress, -description, -homeType, -latest_saledate)

holdoutDF$livingArea_year <- holdoutDF$livingAreaSqFt * holdoutDF$yearBuilt
holdoutDF$lot_pool <- holdoutDF$lotSizeSqFt * holdoutDF$hasPool
holdoutDF$distance_school <- holdoutDF$milesFromDowntown * holdoutDF$avgSchoolRating
holdoutDF$garage_combo <- holdoutDF$garageSpaces * holdoutDF$hasGarage
holdoutDF$area_bedrooms <- holdoutDF$livingAreaSqFt * holdoutDF$numOfBedrooms
holdoutDF$spa_pool <- holdoutDF$hasSpa * holdoutDF$hasPool
holdoutDF$granite_reno <- holdoutDF$graniteCounters * holdoutDF$remodeledOrRenovated
holdoutDF$school_quality_access <- holdoutDF$avgSchoolRating * holdoutDF$avgSchoolDistance
holdoutDF$tax_age <- holdoutDF$propertyTaxRate * holdoutDF$yearBuilt
holdoutDF$bed_bath_ratio <- holdoutDF$numOfBathrooms * holdoutDF$numOfBedrooms

train_zip_cols <- grep("^zipcode_", names(train), value = TRUE)
holdout_zip_cols <- grep("^zipcode_", names(holdoutDF), value = TRUE)
missing_zip_cols <- setdiff(train_zip_cols, holdout_zip_cols)
for (col in missing_zip_cols) {
  holdoutDF[[col]] <- 0
}

rf_preds_holdout <- exp(predict(rf_mtry_1, newdata = holdoutDF))
bag_preds_holdout <- exp(predict(bag_austinhouses, newdata = holdoutDF))
holdoutDF <- holdoutDF[, predictors]
dholdout <- xgb.DMatrix(data = as.matrix(holdoutDF[, predictors]))
xgb_preds_holdout <- exp(predict(xgb_final, newdata = dholdout))
predictions_df <- data.frame(
  XGBoost = xgb_preds_holdout,
  RandomForest = rf_preds_holdout,
  Bagging = bag_preds_holdout
)

head(predictions_df)

cat("Unpruned Tree RMSE:", round(test_rmse, 2))
cat("Pruned Tree RMSE:", round(test_rmse_pruned, 2))
cat("Bagging RMSE:", round(rmse_bag, 2))
cat("Random Forest RMSE:", round(rmse_mtry_1, 2))
cat("XGBoost RMSE:", round(rmse_xgb, 2))

write.csv(
  data.frame(latestPrice_Predicted = xgb_preds_holdout),
  "holdout_predictions_group13.csv",
  row.names = FALSE
)




