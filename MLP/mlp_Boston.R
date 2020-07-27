# INSTALL/ACTIVATE PACKAGES -----------------------------------------------
#install.packages(tensorflow)
#install.packages(remotes)
#remotes::install_github("rstudio/tensorflow")
#install.packages(RSNNS)
#install.packages(ggplot2)

library(tensorflow) # build ANN model
#install_tensorflow(version = "2.0.0")
library(keras) # build ANN model
library(RSNNS) # Data normalisation
library(ggplot2)

# LOAD DATA ---------------------------------------------------------------
library(MASS)
attach(Boston)

# CLEAN DATA --------------------------------------------------------------
dataset <- na.omit(Boston)
with(Boston, sum(is.na(crim)))

# TRAIN/TEST DATA SPLIT ---------------------------------------------------

# Random sampling, create training (80%) and test set (20%)
set.seed(1)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 2:14])
Y_train = as.matrix(dataset[index , 1])
X_test = as.matrix(dataset[-index , 2:14])
Y_test = as.matrix(dataset[-index , 1])

# DATA SCALING ------------------------------------------------------------
# Normalize data, range from 0 to 1
scaled_X_train = normalizeData(X_train, type = "0_1")
scaled_X_test = normalizeData(X_test, type = "0_1")
scaled_Y_train = normalizeData(Y_train, type = "0_1")
scaled_Y_test = normalizeData(Y_test, type = "0_1")

# CONSTRUCT MODELS --------------------------------------------------------

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

# MODELS STRUCTURE SUMMARY ----------------------------------------------------


mlp %>% summary()
mlp_l2 %>% summary()
mlp_l1 %>% summary()
mlp_l1_l2 %>% summary()

# TRAIN/FIT HISTORIES
# K-FOLD CROSS VALIDATION -------------------------------------------------
k <- 5
indices <- sample(1:nrow(scaled_X_train))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 

train_mse_histories_mlp <- NULL
val_mse_histories_mlp <- NULL
train_mse_histories_mlp_l2 <- NULL
val_mse_histories_mlp_l2 <- NULL
train_mse_histories_mlp_l1 <- NULL
val_mse_histories_mlp_l1 <- NULL
train_mse_histories_mlp_l1_l2 <- NULL
val_mse_histories_mlp_l1_l2 <- NULL

for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # TRAINING/VALIDATION DATA PARTITION ----------------------------------
  
  # Prepare the validation data: data from partition # k
  indices_val <- which(folds == i, arr.ind = TRUE) 
  scaled_X_val <- scaled_X_train[indices_val,]
  scaled_Y_val <- scaled_Y_train[indices_val]
  
  # Prepare the training data: data from all other partitions
  partial_scaled_X_train <- scaled_X_train[-indices_val,]
  partial_scaled_Y_train <- scaled_Y_train[-indices_val]
  
  # TRAIN/FIT MODELS  ----------------------------------------------------------
  
  # Create call backs
  set.seed(1)
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 80 == 0) cat("\n")
      cat(".")
    }
  )  
  
  # Create early stopping
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
  
  # Fit models MLP 
  set.seed(1)
  history_mlp <- mlp %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  # Fit models MLP, using L2 regularisation 
  set.seed(1)
  history_mlp_l2 <- mlp_l2 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  # Fit models MLP, using L1 regularisation 
  set.seed(1)
  history_mlp_l1 <- mlp_l1 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))
  
  # Fit models MLP, using L1 and L2 regularisation 
  set.seed(1)
  history_mlp_l1_l2 <- mlp_l1_l2 %>% fit(
    partial_scaled_X_train,
    partial_scaled_Y_train,
    validation_data = list(scaled_X_val, scaled_Y_val),
    epochs = 100,
    verbose = 0, 
    batch_size = 1,
    callbacks = list(early_stop, print_dot_callback))

  # LOAD TRAIN/FIT HISTORIES -----------------------------------------------------
  
  add_train_mse_history_mlp <- history_mlp$loss
  train_mse_histories_mlp <- rbind(train_mse_histories_mlp, 
                                   add_train_mse_history_mlp)
  add_val_mse_histories_mlp <- history_mlp$val_loss
  val_mse_histories_mlp <- rbind(val_mse_histories_mlp, 
                                 add_val_mse_histories_mlp)
  
  add_train_mse_histories_mlp_l2 <- history_mlp_l2$loss
  train_mse_histories_mlp_l2 <- rbind(train_mse_histories_mlp_l2, 
                                      add_train_mse_histories_mlp_l2)
  add_val_mse_histories_mlp_l2 <- history_mlp_l2$val_loss
  val_mse_histories_mlp_l2 <- rbind(val_mse_histories_mlp_l2, 
                                    add_val_mse_histories_mlp_l2)
  
  add_train_mse_histories_mlp_l1 <- history_mlp_l1$loss
  train_mse_histories_mlp_l1 <- rbind(train_mse_histories_mlp_l1, 
                                      add_train_mse_histories_mlp_l1)
  add_val_mse_histories_mlp_l1 <- history_mlp_l1$val_loss
  val_mse_histories_mlp_l1 <- rbind(val_mse_histories_mlp_l1, 
                                    add_val_mse_histories_mlp_l1)
  
  add_train_mse_histories_mlp_l1_l2 <- history_mlp_l1_l2$loss
  train_mse_histories_mlp_l1_l2 <- rbind(train_mse_histories_mlp_l1_l2, 
                                         add_train_mse_histories_mlp_l1_l2)
  add_val_mse_histories_mlp_l1_l2 <- history_mlp_l1_l2$val_loss
  val_mse_histories_mlp_l1_l2 <- rbind(val_mse_histories_mlp_l1_l2, 
                                       add_val_mse_histories_mlp_l1_l2)
  
}
# MODELS COMPARISION -----------------------------------------------------------
# Compute the average train/val MSE for all folds of 4 models
d1 <- data.frame(
  train_loss = apply(train_mse_histories_mlp, 2, mean), 
  valid_loss = apply(val_mse_histories_mlp, 2, mean))
plt1 <- plot_Keras_fit_trajectory(
  d1,
  title = "MLP performance by epoch")
suppressWarnings(print(plt1)) # too few points for loess

d2 <- data.frame(
  train_loss = apply(train_mse_histories_mlp_l2, 2, mean), 
  valid_loss = apply(val_mse_histories_mlp_l2, 2, mean))
plt2 <- plot_Keras_fit_trajectory(
  d2,
  title = "MLP (L2 regularisation) performance by epoch")
suppressWarnings(print(plt2)) # too few points for loess

d3 <- data.frame(
  train_loss = apply(train_mse_histories_mlp_l1, 2, mean), 
  valid_loss = apply(val_mse_histories_mlp_l1, 2, mean))
plt3 <- plot_Keras_fit_trajectory(
  d3,
  title = "MLP (L1 regularisation) performance by epoch")
suppressWarnings(print(plt3)) # too few points for loess

d4 <- data.frame(
  train_loss = apply(train_mse_histories_mlp_l1_l2, 2, mean), 
  valid_loss = apply(val_mse_histories_mlp_l1_l2, 2, mean))
plt4 <- plot_Keras_fit_trajectory(
  d4,
  title = "MLP (L1 and L2 regularisation) performance by epoch")
suppressWarnings(print(plt4)) # too few points for loess

# TEST THE BEST MODEL -------------------------------------------------------------
# Predict the normalised y
set.seed(1)
scaled_Y_hat <- predict(mlp, scaled_X_test)

# Invert the normalised y back to the original scale
Y_hat = denormalizeData(scaled_Y_hat, getNormParameters(scaled_Y_test))

# MODEL RESULT ---------------------------------------------------------
# Calculate the MSE
MSE = mean((Y_test - Y_hat)^2)

# Calculate the R^2
R2 = 1 - sum((Y_test - Y_hat)^2)/sum((Y_test - mean(Y_test))^2)

paste("MSE:", MSE, "R squared:", R2)
detach(Boston)
