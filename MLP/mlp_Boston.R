#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION 
  
#install.packages(kerasR)
#install.packages(tensorflow)
#install.packages(httr)
#install.packages(RCurl)
#install.packages(readxl)
#install.packages(remotes)
#install.packages(tidyverse)
#remotes::install_github("rstudio/tensorflow")
library(tensorflow)
#install_tensorflow(version = "2.0.0")
library(keras)
library(kerasR)
library(httr)
library(RCurl)
library(readxl)
library(tidyverse)
#------------------------------------------------------------------------
## 2. DATA PREPROCESSING
# Remove missing values
library(MASS)
attach(Boston)
dataset <- na.omit(Boston)
with(Boston, sum(is.na(crim)))
#------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
# Random sampling, create training (80%) and test set (20%)
set.seed(80)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 2:14])
Y_train = as.matrix(dataset[index , 1])
X_test = as.matrix(dataset[-index , 2:14])
Y_test = as.matrix(dataset[-index , 1])



#------------------------------------------------------------------------
## 4. DATA SCALING
# Normalize training data
scaled_X_train = scale(X_train)

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(scaled_X_train, "scaled:center") 
col_stddevs_train <- attr(scaled_X_train, "scaled:scale")
scaled_X_test <- scale(X_test, center = col_means_train, scale = col_stddevs_train)

#------------------------------------------------------------------------
## 5. TRAIN NN MODEL
# Construct the NN
   
mlp <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = dim(X_train)[2]) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear")

mlp_l2 <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear")

mlp_l1 <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l1(l = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_regularizer = regularizer_l1(l = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear")

mlp_l1_l2 <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear")
opt<-optimizer_adam( lr= 0.001 , decay = 0, clipnorm = 1 )
mlp %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_absolute_error"))
mlp_l2 %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_absolute_error"))
mlp_l1 %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_absolute_error"))
mlp_l1_l2 %>% compile(
  loss = "mse", optimizer = opt,metrics = list("mean_absolute_error"))

mlp %>% summary()
mlp_l2 %>% summary()
mlp_l1 %>% summary()
mlp_l1_l2 %>% summary()

#Train the NN
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 200)
set.seed(80)
visualisation <- mlp %>% fit(
  X_train,
  Y_train,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0, callbacks = list(early_stop)
)

set.seed(80)
visualisation_l2 <- mlp_l2 %>% fit(
  X_train,
  Y_train,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0, callbacks = list(early_stop)
)
set.seed(80)
visualisation_l1 <- mlp_l1 %>% fit(
  X_train,
  Y_train,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0, callbacks = list(early_stop)
)
set.seed(80)
visualisation_l1_l2 <- mlp_l1_l2 %>% fit(
  X_train,
  Y_train,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0, callbacks = list(early_stop)
)

# Visualisation
library(ggplot2)
plot(visualisation)
plot(visualisation_l2)
plot(visualisation_l1)
plot(visualisation_l1_l2)
#c(loss, MAE) %<-% (model %>% evaluate(X_test, Y_test, verbose = 0))
#paste0("Mean squared error on test set: ", sprintf("%.2f", loss), "%")
mlp %>% evaluate(X_test, Y_test)
mlp_l2 %>% evaluate(X_test, Y_test)
mlp_l1 %>% evaluate(X_test, Y_test)
mlp_l1_l2 %>% evaluate(X_test, Y_test)

#------------------------------------------------------------------------
## 6. TEST NN MODEL
Y_test_pred <- mlp %>% predict(X_test)
Y_test_pred_l2 <- mlp_l2() %>% predict(X_test)
Y_test_pred_l1 <- mlp_l1() %>% predict(X_test)
Y_test_pred_l1_l2 <- mlp_l1_l2() %>% predict(X_test)

#------------------------------------------------------------------------
## 7. K-FOLD CROSS VALIDATION

## 8. MODEL ASSESSMENT: performance, accuracy, interpretability, efficiency.


