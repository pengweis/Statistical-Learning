------------------------------------------------------------------------
## 1. PACKAGES INSTALLATION AND ACTIVATION 
  
#install.packages(keras)
#install.packages(tensorflow)
#install.packages(MASS)
#install.packages("neuralnet")
library(keras)
library(tensorflow)
library(MASS)
library(neuralnet)

------------------------------------------------------------------------
## 2. DATA PREPROCESSING
# Remove missing values
attach(Boston)
Boston <- na.omit(Boston)
with(Boston, sum(is.na(crim)))

------------------------------------------------------------------------
## 3. DATA SPLIT: TRAIN DATASET AND TEST DATASET 
# Random sampling, create training (80%) and test set (20%)
set.seed(80)
samplesize = 0.80 * nrow(Boston)
index = sample( seq_len ( nrow ( Boston ) ), size = samplesize )
trainNN = Boston[index , ]
testNN = Boston[-index , ]

------------------------------------------------------------------------
## 4. DATA SCALING
# Data scaling by min-max normalization 
max = apply(Boston , 2 , max)
min = apply(Boston, 2 , min)
scaled = as.data.frame(scale(Boston, center = min, scale = max - min))
train_scaled_NN = scaled[index , ]
test_scaled_NN = scaled[-index , ]

------------------------------------------------------------------------
## 5. TRAIN NN MODEL
set.seed(2)
NN = neuralnet(crim ~ zn + indus + chas + nox + rm + age  + dis + rad + tax + 
                 ptratio + black + lstat + medv, train_scaled_NN, hidden = 5 ,
               linear.output = TRUE)
NN$result.matrix
plot(NN)

------------------------------------------------------------------------
## 6. TEST NN MODEL
predict_test_scaled_NN = compute(NN, test_scaled_NN[,c(2:14)])
predict_test_scaled_NN <- data.frame(actual = test_scaled_NN$crim, 
                                prediction = predict_test_scaled_NN$net.result)

------------------------------------------------------------------------
## 7. MODEL ACCURACY
# Revert scaled data back to originally formatted data  
predict_test_NN=predict_test_scaled_NN$prediction * abs(diff(range(crim))) + min(crim)
actual_test_NN=predict_test_scaled_NN$actual * abs(diff(range(crim))) + min(crim)

# Regression: RMSE and R^2 for the test set
RMSE = ( sum((actual_test_NN-predict_test_NN)^2) / nrow(testNN) )^ 0.5
R_squared = 1 - 
  sum((actual_test_NN-predict_test_NN)^2)/sum((actual_test_NN - mean(actual_test_NN))^2)
print('RMSE = ')  
RMSE
print('R squared = ')  
R_squared

# Create the data arguments for glmnet
x = model.matrix (crim ~., Boston) [,1]
y = Boston$crim
dim(x)
dim(y)

# Ridge regression regularisation
#install.packages(glmnet)
library (glmnet)
grid =10^ seq (10 , -2 , length =100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge.mod))

