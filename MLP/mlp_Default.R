#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION
#------------------------------------------------------------------------
#install.packages(tensorflow)
#install.packages(httr)
#install.packages(RCurl)
#install.packages(readxl)
#install.packages(remotes)
#remotes::install_github("rstudio/tensorflow")
#install.packages(ggplot2)
#install.packages(matrixStats)
#install.packages(RSNNS)

library(tensorflow)
#install_tensorflow(version = "2.0.0")
library(keras)
library(httr)
library(RCurl)
library(readxl)
library(matrixStats)
library(RSNNS)

#------------------------------------------------------------------------
## 2. DATA PREPROCESSING
#------------------------------------------------------------------------
# Get data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
GET(url, write_disk("default.xls", overwrite=TRUE))
dataset <- read_xls('default.xls', sheet = 1, skip = 1)
# Drop the index column and rename the target variable column 
dataset$ID <- NULL
colnames(dataset)[24] <- "default"
# Remove missing values
dataset <- na.omit(dataset)
with(dataset, sum(is.na(default)))

#------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
#------------------------------------------------------------------------
# Random sampling, create training (80%) and test set (20%)
set.seed(1)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 1:23])
Y_train = as.matrix(dataset[index , 24])
X_test = as.matrix(dataset[-index , 1:23])
Y_test = as.matrix(dataset[-index , 24])

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
mlp <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0)) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.0001)) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l2(l = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))
  
set.seed(1)
mlp_l1 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.0001)) %>% 
  #layer_dropout(rate = 0.5)%>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1(l = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
mlp_l1_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001 )) %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 8, activation = 'relu', 
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0),
              kernel_regularizer = regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001))%>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              kernel_initializer = "orthogonal",
              bias_initializer = initializer_constant(0))

set.seed(1)
opt<-optimizer_adam( lr= 0.00001 , decay = 0, clipnorm = 1 )
mlp %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c(precision=metric_precision, ))
mlp_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy", "Precision", "Recall"))
mlp_l1 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy", "Precision", "Recall"))
mlp_l1_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy", "Precision", "Recall"))

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
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# set class weight for the imbalanced dataset
total = nrow(dataset)
neg = binCounts(dataset$default, bx =c(0, 1))
pos = total - neg

set.seed(1)
visualisation <- mlp %>% 
  fit(X_train, 
      Y_train, 
      epochs = 100,
      verbose = 0, 
      validation_split = 0.2, 
      shuffle = TRUE,   
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))

set.seed(1)
visualisation_l2 <- mlp_l2 %>% 
  fit(X_train, Y_train, epochs = 100,
      verbose = 0, validation_split = 0.2, shuffle = TRUE,
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))

set.seed(1)
visualisation_l1 <- mlp_l1 %>% 
  fit(X_train, Y_train, epochs = 100,
      verbose = 0, validation_split = 0.2, shuffle = TRUE,  
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))

set.seed(1)
visualisation_l1_l2 <- mlp_l1_l2 %>% 
  fit(X_train, Y_train, epochs = 100,
      verbose = 0, validation_split = 0.2, shuffle = TRUE,
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
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

#Evaluate the model
set.seed(1)
mlp %>% evaluate(X_test, Y_test)
set.seed(1)
mlp_l2 %>% evaluate(X_test, Y_test)
set.seed(1)
mlp_l1 %>% evaluate(X_test, Y_test)
set.seed(1)
mlp_l1_l2 %>% evaluate(X_test, Y_test)

#------------------------------------------------------------------------
## 6. TEST NN MODEL
#------------------------------------------------------------------------
set.seed(1)
pred_mlp <- predict_classes(mlp, X_test)
table(Y_test, pred_mlp)
mean(Y_test == pred_mlp)

set.seed(1)
pred_mlp_l2 <- predict_classes(mlp_l2, X_test)
table(Y_test, pred_mlp_l2)
mean(Y_test == pred_mlp_l2)

set.seed(1)
pred_mlp_l1 <- predict_classes(mlp_l1, X_test)
table(Y_test, pred_mlp_l1)
mean(Y_test == pred_mlp_l1)

set.seed(1)
pred_mlp_l1_l2 <- predict_classes(mlp_l1_l2, X_test)
table(Y_test, pred_mlp_l1_l2)
mean(Y_test == pred_mlp_l1_l2)

df <- data.frame("Actual" = Y_test, "Baseline" = pred_mlp, "L2" = pred_mlp_l2, "L1" = pred_mlp_l1, "L1 and L2" = pred_mlp_l1_l2)
head(df[1:5], 10)
summary(df)

#------------------------------------------------------------------------
## 7. MODEL ASSESSMENT: performance, accuracy, interpretability, efficiency.
#------------------------------------------------------------------------



