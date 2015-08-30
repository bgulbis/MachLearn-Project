# Weight Lifting Exercise Prediction
Brian Gulbis  
`r format(Sys.Date(), "%B %d, %Y")`  

## Introduction

In this project, the Human Activity Recognition (HAR) data set will be used to build a model which predicts the manner in which an exercise was performed.  









## Data Processing

### Create Cross-Validation Partition

The training data will be split into a training set and a testing set which will be used to fit the model and estimate the out of sample error.


```r
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
train.data <- training[inTrain,]
valid.data <- training[-inTrain,]
```

## Exploratory Analysis


```r
#summary(training)

#featurePlot(x=training[,c("avg_roll_belt","var_roll_belt")], y=training$classe, plot="pairs")

#qplot(var_roll_belt, classe, data=training)
```

### Near Zero Frequency

Remove near-zero variables


```r
train.set <- select(train.data, -(X:new_window), -classe)
near.zero <- nearZeroVar(train.set)
near.zero.fields <- nearZeroVar(train.set, saveMetrics=TRUE)
train.set <- train.set[, -near.zero]
```

### Find Highly-Correlated Predictors

Find highly-correlated predictors and remove


```r
train.cor <- cor(train.set, use="complete.obs")
high.cor <- findCorrelation(train.cor, cutoff = 0.8)
train.set <- train.set[, -high.cor]
```

### Center and Scale; Impute Missing Values

Use k-nearest neighbor to impute missing values


```r
set.seed(1235)
pre.proc <- preProcess(train.set, method=c("knnImpute"))
train.set <- predict(pre.proc, train.set)
train.set$classe <- train.data$classe
```

### Prep Training Validation Set


```r
valid.set <- select(valid.data, -(X:new_window), -classe)
valid.set <- valid.set[, -near.zero]
valid.set <- valid.set[, -high.cor]
valid.set <- predict(pre.proc, valid.set)
valid.set$classe <- valid.data$classe
```

### Prep Final Test Set


```r
test.set <- dplyr::select(testing, -(X:new_window), -problem_id)
test.set <- test.set[, -near.zero]
test.set <- test.set[, -high.cor]
test.set <- predict(pre.proc, test.set)
```







## Prediction Models

Use Random Forest to create predication model. This model was chosen by first fitting several models using different techniques (Boosting, LDA, and creating a stacked model) and comparing the out of sample error using the training validation set. The model created by the Random Forest method had the highest accuracy, and therefore will be used to predict on the final testing set.


```r
trCtrl <- trainControl(method="repeatedcv", seeds=myseeds)

modelRF.save <- "modelRF.Rds"

if (file.exists(modelRF.save)) {
    # Read the model in and assign it to a variable.
    library(randomForest)
    modelRF <- readRDS(modelRF.save)
} else {
    # Otherwise, run the training.
    modelRF <- train(classe ~ ., method="rf", data=train.set, trControl=trCtrl)
    saveRDS(modelRF, modelRF.save)
}
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
predRF <- predict(modelRF, valid.set)
cmRF <- confusionMatrix(predRF, valid.set$classe)
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

Out of sample error is 0.0053018

## Predict Testing Set


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```


```r
predTest <- predict(modelRF, test.set)
answer <- as.character(predTest)
pml_write_files(answer)
saveRDS(predTest, "test_predictions.Rds")
```

