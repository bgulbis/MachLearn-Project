# Weight Lifting Exercise Prediction
Brian Gulbis  
`r format(Sys.Date(), "%B %d, %Y")`  

## Introduction

In the Weight Lifting Exercises data set, six healthy subjects were asked to perform sets of dumbbell biceps curls in five different manners: 

1. Exactly according to specifications
2. Throwing the elbows to the front
3. Lifting the dumbbell only halfway
4. Lowering the dumbbell only halfway
5. Throwing the hips to the front


Data were collected from sensors on the belt, the arm, the forearm, and the dumbbell for each subject as they performed the exercises using these five techniques. Using the accelerometer, gyroscope, and magnetometer readings from the four sensors, eight features were calculated: mean, variance, standard deviation, max, min, amplitude, kurtosis, skewness. 

In this project, the calculated features of the data set will be used to build a model which predicts the manner in which an exercise was performed. 









## Creating a Cross-Validation Partition

A training data set and a test data set have been provided. To develop the prediction model, cross-validation will be used. The training data set will be split into a training set containing 75% of the original training set, and a validation set containing 25% of the original training set. The new training set will be used to fit different models and determine which model results in the best prediction estimate, while the validation set will be used to estimate the out of sample error.


```r
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
train.data <- training[inTrain,]
valid.data <- training[-inTrain,]
```



## Pre-Processing the Training Data

The first step in pre-processing the new training data set will be to remove any near-zero covariates. The next step will be to find and remove any highly-correlated predictors. The final pre-processing steps will be to center and scale the data, and impute missing values using the k-nearest neighbor method.

The same pre-processing will be applied to the training validation and final testing sets, using the results from the near zero, correlation, and imputation processes which were applied to the training subset. 



```r
train.set <- select(train.data, -(X:new_window), -classe)
near.zero <- nearZeroVar(train.set)
near.zero.fields <- nearZeroVar(train.set, saveMetrics=TRUE)
train.set <- train.set[, -near.zero]

train.cor <- cor(train.set, use="complete.obs")
high.cor <- findCorrelation(train.cor, cutoff = 0.8)
train.set <- train.set[, -high.cor]

set.seed(1235)
pre.proc <- preProcess(train.set, method=c("knnImpute"))
train.set <- predict(pre.proc, train.set)
train.set$classe <- train.data$classe
```










## Building the Predication Model

The final prediction model was selected by first fitting several models using three different techniques (Random Forest, Boosting, Linear Discriminant Analysis) as well as a fourth stacked model, which included these three models. The out of sample error for each model was then calculated using the training validation set. The model created by the Random Forest method had the highest accuracy, and therefore will be used to predict on the final testing set. Looking at the results of this model applied to the training subset, the estimated error rate is 0.33%. To more accurately determine the true out of sample error rate, the model will be applied to the validation portion of the train data set. 




```r
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 43
## 
##         OOB estimate of  error rate: 0.33%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    2    0    0    1 0.0007168459
## B    6 2836    6    0    0 0.0042134831
## C    0   13 2554    0    0 0.0050642774
## D    0    0   14 2396    2 0.0066334992
## E    0    1    0    4 2701 0.0018477458
```

### Out of Sample Error




```r
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    3    0    0    0
##          B    2  944    1    0    0
##          C    0    2  853   11    0
##          D    0    0    1  789    2
##          E    0    0    0    4  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9922, 0.9965)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9947   0.9977   0.9813   0.9978
## Specificity            0.9991   0.9992   0.9968   0.9993   0.9990
## Pos Pred Value         0.9979   0.9968   0.9850   0.9962   0.9956
## Neg Pred Value         0.9994   0.9987   0.9995   0.9964   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1925   0.1739   0.1609   0.1833
## Detection Prevalence   0.2847   0.1931   0.1766   0.1615   0.1841
## Balanced Accuracy      0.9989   0.9970   0.9972   0.9903   0.9984
```

After applying the model to the validation subset, the out of sample error for the final model using the Random Forest method is estimated at 0.53%

## Predict Testing Set

The final model was then used to predict the manner in which the exercise was performed for the 20 values contained in the test set. 





