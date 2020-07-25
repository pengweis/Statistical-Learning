#------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION

#install.packages(kerasR)
#install.packages(tensorflow)
#install.packages(httr)
#install.packages(RCurl)
#install.packages(readxl)
#install.packages(remotes)
#remotes::install_github("rstudio/tensorflow")
#install.packages(ggplot2)
#install.packages(matrixStats)

library(tensorflow)
#install_tensorflow(version = "2.0.0")
library(keras)
library(kerasR)
library(httr)
library(RCurl)
library(readxl)
library(ggplot2)
library(matrixStats)

#------------------------------------------------------------------------
## 2. DATA PREPROCESSING
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
# Explore the data

#------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
# Random sampling, create training (80%) and test set (20%)
set.seed(80)
samplesize = 0.80 * nrow(dataset)
index = sample( seq_len ( nrow ( dataset ) ), size = samplesize )
X_train = as.matrix(dataset[index , 1:23])
Y_train = as.matrix(dataset[index , 24])
X_test = as.matrix(dataset[-index , 1:23])
Y_test = as.matrix(dataset[-index , 24])

#------------------------------------------------------------------------
## 4. DATA SCALING 

scaled_X_train = scale(X_train)
scaled_X_test = scale(X_test)

#------------------------------------------------------------------------
## 5. TRAIN NN MODEL
# Construct the NN
set.seed(80)
mlp <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2]) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = 'relu')%>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

set.seed(80)
mlp_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = 'relu', 
              kernel_regularizer = regularizer_l2(l = 0.001))%>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')
  
set.seed(80)
mlp_l1 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l1(l = 0.001)) %>% 
  layer_dropout(rate = 0.5)%>% 
  layer_dense(units = 4, activation = 'relu', 
              kernel_regularizer = regularizer_l1(l = 0.001))%>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

set.seed(80)
mlp_l1_l2 <- keras_model_sequential()  %>% 
  layer_dense(units = 16, input_shape = dim(X_train)[2], 
              kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001 )) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 4, activation = 'relu', 
              kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001))%>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

opt<-optimizer_adam( lr= 0.0001 , decay = 0, clipnorm = 1 )
mlp %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l1 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))
mlp_l1_l2 %>% 
  compile(loss = 'binary_crossentropy', optimizer = opt, metrics = c("accuracy"))

mlp %>% summary()
mlp_l2 %>% summary()
mlp_l1 %>% summary()
mlp_l1_l2 %>% summary()

#Train the NN
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)  
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# set class weight for the imbalanced dataset
set.seed(80)
total = nrow(dataset)
neg = binCounts(dataset$default, bx =c(0, 1))
pos = total - neg
visualisation <- mlp %>% 
  fit(X_train, Y_train,
          batch_size = 128, epochs = 100,
          verbose = 0, validation_split = 0.2,   
          class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
          callbacks = list(early_stop, print_dot_callback))
plot(visualisation, smooth = FALSE, main = "Original MultiLayer Perceptron", xlab = "Number of Iterations")

visualisation_l2 <- mlp_l2 %>% 
  fit(X_train, Y_train,
      batch_size = 128, epochs = 100,
      verbose = 0, validation_split = 0.2,  
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))
plot(visualisation_l2, smooth = FALSE, main = "L2 Regularised MultiLayer Perceptron", xlab = "Number of Iterations") 

visualisation_l1 <- mlp_l1 %>% 
  fit(X_train, Y_train,
      batch_size = 128, epochs = 100,
      verbose = 0, validation_split = 0.2,  
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))
plot(visualisation_l1, smooth = FALSE, main = "L1 Regularised MultiLayer Perceptron", xlab = "Number of Iterations")

visualisation_l1_l2 <- mlp_l1_l2 %>% 
  fit(X_train, Y_train,
      batch_size = 128, epochs = 100,
      verbose = 0, validation_split = 0.2,  
      class_weight = list("0"=total/(2*neg),"1"=total/(2*pos)),
      callbacks = list(early_stop, print_dot_callback))
plot(visualisation_l1_l2, smooth = FALSE, main = "L1 and L2 Regularised MultiLayer Perceptron", xlab = "Number of Iterations")

#Evaluate the model
set.seed(80)
mlp %>% evaluate(X_test, Y_test, batch_size = 32)
mlp_l2 %>% evaluate(X_test, Y_test, batch_size = 32)
mlp_l1 %>% evaluate(X_test, Y_test, batch_size = 32)
mlp_l1_l2 %>% evaluate(X_test, Y_test, batch_size = 32)

#------------------------------------------------------------------------
## 6. TEST NN MODEL
set.seed(80)
pred_mlp <- predict_classes(mlp, X_test)
pred_mlp_l2 <- predict_classes(mlp_l2, X_test)
pred_mlp_l1 <- predict_classes(mlp_l1, X_test)
pred_mlp_l1_l2 <- predict_classes(mlp_l1_l2, X_test)
df <- data.frame(pred_mlp,pred_mlp_l2,pred_mlp_l1,pred_mlp_l1_l2)
summary(df)
#------------------------------------------------------------------------
## 7. K-FOLD CROSS VALIDATION

## 8. MODEL ASSESSMENT: performance, accuracy, interpretability, efficiency.

