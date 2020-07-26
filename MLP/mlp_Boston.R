#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION 
#------------------------------------------------------------------------
#install.packages(kerasR)
#install.packages(tensorflow)
#install.packages(httr)
#install.packages(RCurl)
#install.packages(readxl)
#install.packages(remotes)
#install.packages(tidyverse)
#remotes::install_github("rstudio/tensorflow")
#install.packages(RSNNS)

library(tensorflow)
#install_tensorflow(version = "2.0.0")
library(keras)
library(httr)
library(RCurl)
library(readxl)
library(tidyverse)
library(RSNNS)

#------------------------------------------------------------------------
## 2. DATA PREPROCESSING
#------------------------------------------------------------------------
# Remove missing values
library(MASS)
attach(Boston)
dataset <- na.omit(Boston)
with(Boston, sum(is.na(crim)))
#------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
#------------------------------------------------------------------------
# Random sampling, create training (80%) and test set (20%)
set.seed(1)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 2:14])
Y_train = as.matrix(dataset[index , 1])
X_test = as.matrix(dataset[-index , 2:14])
Y_test = as.matrix(dataset[-index , 1])

#------------------------------------------------------------------------
## 4. DATA SCALING
#------------------------------------------------------------------------
scaled_X_train = normalizeData(X_train, type = "0_1")
scaled_X_test = normalizeData(X_test, type = "0_1")
scaled_Y_train = normalizeData(Y_train, type = "0_1")
scaled_Y_test = normalizeData(Y_test, type = "0_1")

#------------------------------------------------------------------------
## 5. TRAIN NN MODEL
#------------------------------------------------------------------------
# Construct the NN

set.seed(1)
mlp <- keras_model_sequential() %>%
  layer_dense(units = 8, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l2 <- keras_model_sequential() %>%
  layer_dense(units = 8, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear",
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l1 <- keras_model_sequential() %>%
  layer_dense(units = 8, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l1_l2 <- keras_model_sequential() %>%
  layer_dense(units = 8, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.003, l2 = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = "relu", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.003, l2 = 0.003)) %>%
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "linear", 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
opt<-optimizer_adam( lr= 0.003 , decay = 0, clipnorm = 1 )

mlp %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_squared_error"))
mlp_l2 %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_squared_error"))
mlp_l1 %>% compile(
  loss = "mse", optimizer = opt, metrics = list("mean_squared_error"))
mlp_l1_l2 %>% compile(
  loss = "mse", optimizer = opt,metrics = list("mean_squared_error"))

mlp %>% summary()
mlp_l2 %>% summary()
mlp_l1 %>% summary()
mlp_l1_l2 %>% summary()

#Train the NN
set.seed(1)
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)  
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 800)

set.seed(1)
visualisation <- mlp %>% fit(
  scaled_X_train,
  scaled_Y_train,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0, 
  shuffle = TRUE,   
  callbacks = list(early_stop, print_dot_callback))


set.seed(1)
visualisation_l2 <- mlp_l2 %>% fit(
  scaled_X_train,
  scaled_Y_train,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0, 
  shuffle = TRUE,   
  callbacks = list(early_stop, print_dot_callback))

set.seed(1)
visualisation_l1 <- mlp_l1 %>% fit(
  scaled_X_train,
  scaled_Y_train,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0, 
  shuffle = TRUE,   
  callbacks = list(early_stop, print_dot_callback))

set.seed(1)
visualisation_l1_l2 <- mlp_l1_l2 %>% fit(
  scaled_X_train,
  scaled_Y_train,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0, 
  shuffle = TRUE,   
  callbacks = list(early_stop, print_dot_callback))

# Visualisation
plot(visualisation, smooth = FALSE, 
     main = "Baseline MultiLayer Perceptron", 
     xlab = "Number of Iterations")
plot(visualisation_l2, smooth = FALSE, 
     main = "L2 Regularised MultiLayer Perceptron", 
     xlab = "Number of Iterations") 
plot(visualisation_l1, smooth = FALSE, 
     main = "L1 Regularised MultiLayer Perceptron", 
     xlab = "Number of Iterations")
plot(visualisation_l1_l2, smooth = FALSE, 
     main = "L1 and L2 Regularised MultiLayer Perceptron", 
     xlab = "Number of Iterations")

#------------------------------------------------------------------------
## 6. TEST NN MODEL
#------------------------------------------------------------------------
set.seed(1)
scaled_Yhat_mlp <- predict(mlp, scaled_X_test)
Yhat_mlp = denormalizeData(scaled_Yhat_mlp, getNormParameters(scaled_Y_test))

set.seed(1)
scaled_Yhat_mlp_l2 <- predict(mlp_l2, scaled_X_test)
Yhat_mlp_l2 = denormalizeData(scaled_Yhat_mlp_l2, getNormParameters(scaled_Y_test))

set.seed(1)
scaled_Yhat_mlp_l1 <- predict(mlp_l1, scaled_X_test)
Yhat_mlp_l1 = denormalizeData(scaled_Yhat_mlp_l1, getNormParameters(scaled_Y_test))

set.seed(1)
scaled_Yhat_mlp_l1_l2 <- predict(mlp_l1_l2, scaled_X_test)
Yhat_mlp_l1_l2 = denormalizeData(scaled_Yhat_mlp_l1_l2, getNormParameters(scaled_Y_test))

df <- data.frame("Actual" = Y_test, "Baseline" = Yhat_mlp, "L2" = Yhat_mlp_l2, "L1" = Yhat_mlp_l1, "L1_L2" = Yhat_mlp_l1_l2)
head(df[1:5], 10)
summary(df)

#Evaluate the model
MSE_mlp = mean((df$Actual - df$Baseline)^2)
MSE_mlp_l2 = mean((df$Actual - df$L2)^2)
MSE_mlp_l1 = mean((df$Actual - df$L1)^2)
MSE_mlp_l1_l2 = mean((df$Actual - df$L1_L2)^2)
sprintf("MSE of MLP on Test dataset: %2f", MSE_mlp)
sprintf("MSE of MLP, using L2 regularisation, on Test dataset: %2f", MSE_mlp_l2)
sprintf("MSE of MLP, using L1 regularisation, on Test dataset: %2f", MSE_mlp_l1)
sprintf("MSE of MLP, using L1 and L2 regularisation, on Test dataset: %2f", MSE_mlp_l1_l2)

#------------------------------------------------------------------------
## 7. K-FOLD CROSS VALIDATION
#------------------------------------------------------------------------

#------------------------------------------------------------------------
## 8. MODEL ASSESSMENT: performance, accuracy, interpretability, efficiency.
#------------------------------------------------------------------------


detach(Boston)
