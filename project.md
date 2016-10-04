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

The final prediction model was selected by first fitting several models using three different techniques (Random Forest, Boosting, Linear Discriminant Analysis) as well as a fourth stacked model, which included the Random Forest and Boosting models, and used Random Forest to create the model. 












```r
trCtrl <- trainControl(method="repeatedcv", seeds=myseeds)

modelRF <- train(classe ~ ., method="rf", data=train.set, trControl=trCtrl)
predRF <- predict(modelRF, valid.set)

modelGBM <- train(classe ~ ., method="gbm", data=train.set, verbose=FALSE, trControl=trCtrl)
predGBM <- predict(modelGBM, valid.set)

modelLDA <- train(classe ~ ., method="lda", data=train.set, trControl=trCtrl)
predLDA <- predict(modelLDA, valid.set)

predDF <- data.frame(predRF, predGBM, classe=valid.set$classe)
modStack <- train(classe ~ ., data=predDF, method="rf")
predStack <- predict(modStack, valid.set)
```

### Out of Sample Error Estimate

The out of sample error for each model was then estimated by creating a Confusion Matrix using the training validation set. The Stacked Model and Random Forest Model had the lowest, and nearly identical, out of sample errors (see table). Since the Stacked Model had the lowest out of sample error, it will be used as the final model to predict on the testing set.

Model | Out of Sample Error
------|------------------------------
Random Forest | 0.53%
Boosting | 1.65%
Linear Discriminate Analysis | 29.34%
Stacked | 0.51%

## Predict on the Testing Set

The final model was then used to predict the manner in which the exercise was performed for the 20 values contained in the test set. 




```r
predTest <- predict(modelStack, test.set)
answer <- as.character(predTest)
pml_write_files(answer)
saveRDS(predTest, "test_predictions.Rds")
```

