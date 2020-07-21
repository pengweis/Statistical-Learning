#install.packages(keras)
library(keras)
#install.packages(tensorflow)
library(tensorflow)

#install.packages(MASS)
library(MASS)
attach(Boston)

# Descriptive statistic
descriptive_stats <- function(Boston){
  str(Boston)
  summary(Boston)
}

# Random sampling, create training (90%) and test set (10%)
samplesize = 0.80 * nrow(Boston)
set.seed(80)
index = sample( seq_len ( nrow ( Boston ) ), size = samplesize )
datatrain = Boston[ index, ]
datatest = Boston[ -index, ]

# Remove missing values
Boston <- na.omit(Boston)
with(Boston, sum(is.na(crim)))

# Create the data arguments for glmnet
x = model.matrix (crim ~., Boston) [,1]
y = Boston$crim

# Ridge regression regularisation
#install.packages(glmnet)
library (glmnet)
grid =10^ seq (10 , -2 , length =100)
ridge.mod = glmnet(x, y, alpha = 0, lambda = grid)
return(ridge.mod) 

# Data scaling by min-max normalization 
max = apply(Boston , 2 , max)
min = apply(Boston, 2 , min)
scaled = as.data.frame(scale(Boston, center = min, scale = max - min))
trainNN = scaled[index , ]
testNN = scaled[-index , ]

## Fit neural network 
#install.packages("neuralnet")
library(neuralnet)
set.seed(2)
NN = neuralnet(crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + medv, trainNN, hidden = 5 , linear.output = T )
NN

# plot neural network
plot(NN)

## Prediction using neural network
predict_testNN = compute(NN, testNN[,c(2:14)])
predict_testNN = (predict_testNN$net.result * (max(Boston$crim) - min(Boston$crim))) + min(Boston$crim)
predict_testNN
plot(datatest$crim, predict_testNN, col='red', pch=16, 
     ylab = "Predicted Crime Rate NN", xlab = "real Crime Rate", main="Real Crime Rate vs Predict NN")

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((datatest$crim - predict_testNN)^2) / nrow(datatest)) ^ 0.5
RMSE.NN

## Cross validation of neural network model
# install relevant libraries
#install.packages("boot")
#install.packages("plyr")

# Load libraries
library(boot)
library(plyr)

# Initialize variables
set.seed(50)
k = 100
RMSE.NN = NULL

List = list( )

# Fit neural network model within nested for loop
for(j in 10:65){
  for (i in 1:k) {
    index = sample(1:nrow(Boston),j )
    
    trainNN = scaled[index,]
    testNN = scaled[-index,]
    datatest = Boston[-index,]
    
    NN = neuralnet(crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + medv, trainNN, hidden = 5, linear.output= T)
    predict_testNN = compute(NN,testNN[,c(2:14)])
    predict_testNN = (predict_testNN$net.result*(max(Boston$crim)-min(Boston$crim)))+min(Boston$crim)
    
    RMSE.NN [i]<- (sum((datatest$rating - predict_testNN)^2)/nrow(datatest))^0.5
  }
  List[[j]] = RMSE.NN
}

Matrix.RMSE = do.call(cbind, List)
Matrix.RMSE

## Prepare boxplot
boxplot(Matrix.RMSE[,56], ylab = "RMSE", main = "RMSE BoxPlot (length of traning set = 65)", col="green")

## Variation of median RMSE 
#install.packages("matrixStats")
library(matrixStats)

med = colMedians(Matrix.RMSE)
med

X = seq(10,65)
X
plot (med~X, type = "l", xlab = "Length of Training Set", ylab = "Median RMSE", 
      main = "Variation of RMSE with Length of Training Set", col="blue",lwd=2)

