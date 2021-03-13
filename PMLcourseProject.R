library(ggplot2)
library(caret)
library(doParallel)
library(randomForest)
library(e1071)
library(gbm)
library(survival)
library(splines)
library(plyr)
#Reading data file in project directory downloaded from course page after removing Div/0
training <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
#Removing NA and blank columns and storing it separately and using good columsn alone
training <- training[, 6:dim(training)[2]]
treshold <- dim(training)[1] * 0.95
goodcols <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)
training <- training[, goodcols]
badcols <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, badcols$nzv==FALSE]
training$classe = factor(training$classe)

inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]
#applying same transformation to test data as training
testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodcols]
testing$classe <- NA
testing <- testing[, badcols$nzv==FALSE]
# training with 3 methods random forest, boosting and linear discriminant analysis
#mod1 <- train(classe ~ ., data=training, method="rf")
mod2 <- train(classe ~ ., data=training, method="gbm")
#mod3 <- train(classe ~ ., data=training, method="lda")
#predicting with 3 models with crossvalidation
#pred1 <- predict(mod1, crossv)
pred2 <- predict(mod2, crossv)
#pred3 <- predict(mod3, crossv)
#confusionMatrix(pred1, crossv$classe)
confusionMatrix(pred2, crossv$classe)
#confusionMatrix(pred3, crossv$classe)
#accuracy1 <- sum(pred1 == crossv_test$classe) / length(pred1)
accuracy2 <- sum(pred2 == crossv_test$classe) / length(pred2)
#accuracy3 <- sum(pred3 == crossv_test$classe) / length(pred3)
#predDF <- data.frame(pred1, pred2, pred3, classe=crossv$classe)
#predDF2 <- data.frame(pred1, pred2, classe=crossv$classe)
#combModFit <- train(classe ~ ., method="rf", data=predDF)
#combModFit2 <- train(classe ~ ., method="rf", data=predDF2)
#combPredIn <- predict(combModFit, predDF)
#combPredIn2 <- predict(combModFit2, predDF2)

#getting importance of features in goodcols
varImpRF <- train(classe ~ ., data = training, method = "rf")
varImpObj <- varImp(varImpRF)
plot(varImpObj, main = "Importance of Top 25 Variables", top = 25)
