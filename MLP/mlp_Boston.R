#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION 
#------------------------------------------------------------------------
#install.packages(tensorflow)
#install.packages(remotes)
#remotes::install_github("rstudio/tensorflow")
#install.packages(RSNNS)

library(tensorflow) # build ANN model
#install_tensorflow(version = "2.0.0")
library(keras) # build ANN model
library(RSNNS) # Data normalisation
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
# Normalize data, range from 0 to 1
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

# Create k-fold cross validation
k <- 5
indices <- sample(1:nrow(scaled_X_train))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

mlp_mse_histories <- NULL
l2_mlp_mse_histories <- NULL
l1_mlp_mse_histories <- NULL
l1_l2_mlp_mse_histories <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  indices_val <- which(folds == i, arr.ind = TRUE) 
  scaled_X_val <- scaled_X_train[indices_val,]
  scaled_Y_val <- scaled_Y_train[indices_val]
  
  # Prepare the training data: data from all other partitions
  partial_scaled_X_train <- scaled_X_train[-indices_val,]
  partial_scaled_Y_train <- scaled_Y_train[-indices_val]
  
  #Train the NN
  set.seed(1)
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 80 == 0) cat("\n")
      cat(".")
    }
  )  
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
  
  set.seed(1)
  history_mlp <- mlp %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  set.seed(1)
  history_mlp_l2 <- mlp_l2 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  set.seed(1)
  history_mlp_l1 <- mlp_l1 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  set.seed(1)
  history_mlp_l1_l2 <- mlp_l1_l2 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
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
  
  #Evaluate the model
  MSE_mlp = mean((df$Actual - df$Baseline)^2)
  MSE_mlp_l2 = mean((df$Actual - df$L2)^2)
  MSE_mlp_l1 = mean((df$Actual - df$L1)^2)
  MSE_mlp_l1_l2 = mean((df$Actual - df$L1_L2)^2)
  
  mlp_mse_histories <- rbind(mlp_mse_histories, MSE_mlp)
  l2_mlp_mse_histories <- rbind(l2_mlp_mse_histories, MSE_mlp_l2)
  l1_mlp_mse_histories <- rbind(l1_mlp_mse_histories, MSE_mlp_l1)
  l1_l2_mlp_mse_histories <- rbind(l1_l2_mlp_mse_histories, MSE_mlp_l1_l2)
}


#------------------------------------------------------------------------
## 8. MODEL ASSESSMENT: performance, accuracy, interpretability, efficiency.
#------------------------------------------------------------------------
# Compute the average of the per-epoch MSE scores for all folds
average_mse_history_mlp <- data.frame(
  epoch = seq(1:ncol(mlp_mse_histories)),
  validation_mse = apply(mlp_mse_histories, 2, mean)
)
average_mse_history_mlp_l2 <- data.frame(
  epoch = seq(1:ncol(l2_mlp_mse_histories)),
  validation_mse = apply(l2_mlp_mse_histories, 2, mean)
)
average_mse_history_mlp_l1 <- data.frame(
  epoch = seq(1:ncol(l1_mlp_mse_histories)),
  validation_mse = apply(l1_mlp_mse_histories, 2, mean)
)
average_mse_history_mlp_l1_l2 <- data.frame(
  epoch = seq(1:ncol(l1_l2_mlp_mse_histories)),
  validation_mse = apply(l1_l2_mlp_mse_histories, 2, mean)
)


# Visualisation
library(ggplot2)
ggplot(average_mse_history_mlp, aes(x = epoch, y = validation_mse)) + geom_smooth()
ggplot(average_mse_history_mlp_l2, aes(x = epoch, y = validation_mse)) + geom_smooth()
ggplot(average_mse_history_mlp_l1, aes(x = epoch, y = validation_mse)) + geom_smooth()
ggplot(average_mse_history_mlp_l1_l2, aes(x = epoch, y = validation_mse)) + geom_smooth()



detach(Boston)
