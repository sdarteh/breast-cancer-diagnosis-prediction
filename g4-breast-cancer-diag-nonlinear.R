##### dataset

dim(trainRows)

dim(brecanX_train)
length(brecanY_train)
dim(brecanX_test)
length(brecanY_test)


##### BUILDING MODELS

set.seed(500) 
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                     # summaryFunction = twoClassSummary, #defaultSummary
                     classProbs = TRUE, 
                     savePredictions = TRUE)


## Non-Linear  Classification Models


## 1.1 Quadratic Discriminant Analysis -- QDA
## ************************************************************
install.packages("qda")
library(qda)
library(caret)


# has no tuning...
set.seed(503)
qdaFit <- caret::train(x = brecanX_train, y = brecanY_train,
                        method = "qda", metric = "Kappa", 
                        #tuneGrid = rdaTuneGrid,
                        #preProc = c("center", "scale"),
                        trControl = ctrl)
qdaFit

#predict on test data
qdaPredTest <- predict(qdaFit, newdata = brecanX_test)
qdaPredTest

postResample(qdaPredTest, brecanY_test)

qdaConfMatrix <- confusionMatrix(data = qdaPredTest, reference = brecanY_test)   
qdaConfMatrix

# Predict probs on the test set
# qdaProbPred <- as.numeric(predict(qdaFit, brecanX_test, type = "prob")$M)  #$posterior
qdaProbPred <- predict(qdaFit, brecanX_test, type = "prob")
qdaProbPred_TgtRes <- qdaProbPred$M #qdaProbPred_TgtRes <- qdaProbPred[, 3]

# Create a ROC object
library("pROC") 
qdaROC <- roc(response = brecanY_test, 
              predictor = qdaProbPred_TgtRes,
              levels=rev(levels(brecanY_test)))

print(qdaROC)

# find AUC
qdaAUC <- auc(qdaROC)
print(qdaAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.
plot(qdaFit, 
     main="Quadratic Discriminant Analysis (QDA) Model",
     col=2,lwd=1.3)

plot(qdaFit, plotType = "level",
     main="QDA Model per Tuning Parameters")


#  ROC curve
plot(x = qdaROC, #$predictor, y = qdaROC$response,
     main = "QDA Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col = "deeppink", 
     lty = 1:3, lwd=1.5)







## 1.2 Regularized Discriminant Analysis -- RDA
## ************************************************************
install.packages("rda")
library(rda)
library(caret)

 
rdaTuneGrid <- expand.grid(lambda = seq(0, 1, by = 0.2),
                           gamma = seq(0, 1, by = 0.1))

set.seed(505)
rdaTune <- caret::train(x = brecanX_train, y = brecanY_train,
                        method = "rda", metric = "Kappa", 
                        tuneGrid = rdaTuneGrid,
                        preProc = c("center", "scale"),
                        trControl = ctrl)
rdaTune

#predict on test data
rdaPredTest <- predict(rdaTune, newdata = brecanX_test)
rdaPredTest

postResample(rdaPredTest, brecanY_test)

rdaConfMatrix <- confusionMatrix(data = rdaPredTest, reference = brecanY_test)   
rdaConfMatrix

# Predict probs on the test set
# rdaProbPred <- as.numeric(predict(rdaTune, brecanX_test, type = "prob")$M)  #$posterior
rdaProbPred <- predict(rdaTune, brecanX_test, type = "prob")
rdaProbPred_TgtRes <- rdaProbPred$M #rdaProbPred_TgtRes <- rdaProbPred[, 3]

# Create a ROC object
library("pROC") 
rdaROC <- roc(response = brecanY_test, 
              predictor = rdaProbPred_TgtRes,
              levels=rev(levels(brecanY_test)))

print(rdaROC)

# find AUC
rdaAUC <- auc(rdaROC)
print(rdaAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.
plot(rdaTune, 
     main="Regularized Discriminant Analysis (RDA) Model",
     lwd=1.3)

plot(rdaTune, plotType = "level",
     main="RDA Model per Tuning Parameters")


#  ROC curve
  plot(x = rdaROC, #$predictor, y = rdaROC$response,
       main = "RDA Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = "darkgreen", 
       lty = 1:3, lwd=1.5)
   




## 1.3 Mixture Discriminant Analysis -- MDA
## ************************************************************
library(mda)
library(caret)

## tuning parameters: no. of distributions = subclasses
#mdaCtrl <- trainControl(summaryFunction = multiClassSummary, classProbs = TRUE) 

#the potential subpopulations or clusters within each class
mdaTuneGrid = expand.grid(.subclasses = 1:3) # 3 >> cntDist

set.seed(510)
mdaTune <- caret::train(x = brecanX_train, y = brecanY_train,
                       method = "mda", metric = "Kappa", 
                       tuneGrid = mdaTuneGrid,
                       #preProc = c("center", "scale"),
                       trControl = ctrl)
mdaTune

#predict on test data
mdaPredTest <- predict(mdaTune, newdata = brecanX_test)
mdaPredTest

postResample(mdaPredTest, brecanY_test)

mdaConfMatrix <- confusionMatrix(data = mdaPredTest, reference = brecanY_test)   
mdaConfMatrix

# Predict probs on the test set
# mdaProbPred <- as.numeric(predict(mdaTune, brecanX_test, type = "prob")$M)  #$posterior
mdaProbPred <- predict(mdaTune, brecanX_test, type = "prob")
mdaProbPred_TgtRes <- mdaProbPred$M #mdaProbPred_TgtRes <- mdaProbPred[, 3]


# Create a multiclass ROC curve
library(pROC)
mdaROC <- roc(response = brecanY_test, 
                         predictor = mdaProbPred_TgtRes,
                         levels=rev(levels(brecanY_test)))

print(mdaROC)

# find AUC
mdaAUC <- auc(mdaROC)
print(mdaAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.
plot(mdaTune, 
     main="Mixture Discriminant Analysis (MDA) Model",
     col = 2, lwd=1.3)

# tuning heatmap... 2 params needed
plot(mdaTune, plotType = "level", legacy.axes = TRUE, 
     main="Mixture Discriminant Analysis (MDA) Model: Tuning Heat-map")

#ROC curve
tryCatch({
  
  plot(x = mdaROC, #$predictor, y = mdaROC$response,
       main = "MDA Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = "darkorange", # c("darkorange", "darkgreen", "red"), 
       lty = 1:3, lwd=1.5)
  
  #legend("topright", legend = levels(brecanY_test), 
  #       col = c("darkorange", "darkgreen", "red"),  
  #       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
  #       inset = c(0, -0.08), xpd = TRUE)
  
}, 
error = function(e) {
  print(paste("Error:", e))
})




## 2. Neural Networks model -- NNet
## ************************************************************
## R packages... nnet, RSNNS, qrnn, and neuralnet
install.packages("nnet")
library(nnet)
library(caret)

## tuning parmeters: size[...], decay[...]

##nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
#nnetGrid <- expand.grid(.size = seq(1, 10, by = 2), .decay = c(0.1, 1, 5, 10))

nnetGrid <- expand.grid(.size = 1:10, 
                        .decay = c(0, .1, 1, 2, 4, 6, 8, 11))


#numWts <- (maxSize * (4 + 1)) + ((maxSize+1)*2) ## 4 is the number of predictors ## ((p+1)*H) + ((H+1)*C)
maxSize <- max(nnetGrid$.size) ## maxSize --> H, hinge fxn
cntPred = dim(brecanX_train)[2]
cntClass = length(levels(brecanY_train))
numWts <- (maxSize * (cntPred + 1)) + ((maxSize + 1) * cntClass) ## 4 is the number of predictors

#nnetCtrl <- trainControl(summaryFunction = defaultSummary,classProbs = TRUE) #twoClassSummary
set.seed(330)
nnetTune <- caret::train(x = brecanX_train, y = brecanY_train,
                        method = "nnet", metric = "Kappa",
                        maxit = 2000, MaxNWts = numWts,
                        trace = FALSE,
                        tuneGrid = nnetGrid, 
                        preProcess = c("center", "scale", "spatialSign"),
                        trControl = ctrl)
nnetTune

#predict on test data
nnetPredTest <- predict(nnetTune, newdata = brecanX_test)
nnetPredTest

postResample(nnetPredTest, brecanY_test)

nnetConfMatrix <- confusionMatrix(data = nnetPredTest, reference = brecanY_test)   
nnetConfMatrix

# Predict probs on the test set
# nnetProbPred <- as.numeric(predict(nnetTune, brecanX_test, type = "prob")$M)  #$posterior
nnetProbPred <- predict(nnetTune, brecanX_test, type = "prob")
nnetProbPred_TgtRes <- nnetProbPred$M #nnetProbPred_TgtRes <- nnetProbPred[, 3]

# Create a ROC object
library("pROC") 
nnetROC <- roc(response = brecanY_test, 
                          predictor = nnetProbPred_TgtRes,
                          levels=rev(levels(brecanY_test)))

print(nnetROC)

# find AUC
nnetAUC <- auc(nnetROC)
print(nnetAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.

plot(nnetTune, 
     main="NNet Model",
     lwd=1.3) 

#grid(col = 'lightgrey', lty = 3, lwd = par("lwd"), equilogs = TRUE)

plot(nnetTune, plotType = "level",
     main="NNet Model: Tuning Heat-map")

#  ROC curve
tryCatch({
  
  plot(x = nnetROC, #$predictor, y = nnetROC$response,
       main = "NNet Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = 3, #c("orange", "darkgreen", "red"), 
       lty = 1:3, lwd=1.5)
  
   
},
error = function(e) {
  print(paste("Error:", e))
})







## 3. Flexible Discriminant Analysis model -- FDA
## ************************************************************

library(MASS)
library(mda)
library(earth)
library(caret)

## tuning paramenters: degree [...], nprune [terms retained]... 
## similar to the MARS model
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:30)

#fdaCtrl <- trainControl(method = "cv")
fdaCtrl <- trainControl(method = "cv", number = 10, #repeats = 5, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE, savePredictions = TRUE) 


set.seed(340)
fdaTune <- caret::train(x = brecanX_train, y = brecanY_train,
                        method = "fda", metric = "Kappa",
                        tuneGrid = marsGrid,
                        #preProcess = c("center", "scale"),
                        trControl = fdaCtrl)

fdaTune

#predict on test data
fdaPredTest <- predict(fdaTune, newdata = brecanX_test)
fdaPredTest

postResample(fdaPredTest, brecanY_test)

fdaConfMatrix <- confusionMatrix(data = fdaPredTest, reference = brecanY_test)   
fdaConfMatrix

# Predict probs on the test set
# fdaProbPred <- as.numeric(predict(fdaTune, brecanX_test, type = "prob")$M)  #$posterior
fdaProbPred <- predict(fdaTune, brecanX_test, type = "prob")
fdaProbPred_TgtRes <- fdaProbPred$M #fdaProbPred_TgtRes <- fdaProbPred[, 3]

# Create a ROC object
library("pROC") 
fdaROC <- roc(response = brecanY_test, 
                         predictor = fdaProbPred_TgtRes,
                         levels=rev(levels(brecanY_test)))

print(fdaROC)

# find AUC
fdaAUC <- auc(fdaROC)
print(fdaAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.
## degree = 2 and nprune = 3.
optMdlDeg = 1 #optimal model from training 
optMdlTerm = 18 #optimal model from training 
plot(fdaTune, 
     main=paste(paste(paste("FDA Model: Degree =", optMdlDeg), 
                      " and nprune ="), optMdlTerm), 
     lwd=1.3, col=c(2,4))


#  ROC curve
tryCatch({
  plot(x = fdaROC, #$predictor, y = fdaROC$response,
       main = "FDA Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = 3, #c("orange", "darkgreen", "red"), 
       lty = 1:3, lwd=1.5)
   
  
},
error = function(e) {
  print(paste("Error:", e))
})



plot(fdaTune, plotType = "level",
     main="FDA Model per Tuning Parameters")





## 4. Support Vector Machines model -- SVM
## ************************************************************

## R packages for SVM and other kernels: e1071, kernlab, klaR, and svmPath

library(MASS)
library(kernlab)
library(caret)

#ctrl <- trainControl(summaryFunction = defaultSummary, classProbs = TRUE)

## sigest estimates the range of values for the sigma parameter 
sigmaRangeReduced <- sigest(as.matrix(brecanX_train))

## SVM (ksvm). The estimation is based upon the 0.1 and 0.9 quantile of ||x -x'||^2.
## Basically any value in between those two bounds will produce good results.
## Given a range of values for the "sigma" inverse width parameter in the Gaussian Radial Basis kernel for use with SVM

## log 2 base
minAbsSSE <- -4
maxAbsSSE <- 8 # changed from 6 to 8
#svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1], .C = 2^(seq(-4, 6)))
svmRGridReduced <- expand.grid(.sigma = min(sigmaRangeReduced), 
                               .C = 2^(seq(minAbsSSE, maxAbsSSE))) # minimum val of ||x -x'||^2

library(caret)
set.seed(350)
svmRTune <- caret::train(x = brecanX_train, 
                         y = brecanY_train,
                         method = "svmRadial",
                         metric = "Kappa",
                         preProc = c("center", "scale"),
                         tuneGrid = svmRGridReduced,
                         fit = FALSE,
                         trControl = ctrl)
svmRTune 

## When the outcome is a factor, the function automatically uses prob.model = TRUE. 
## Other kernel functions can be defined via the kernel and kpar arguments.
#library(kernlab)

#predict on test data
svmRPredTest <- predict(svmRTune, newdata = brecanX_test)
svmRPredTest

postResample(svmRPredTest, brecanY_test)

svmConfMatrix <- confusionMatrix(data = svmRPredTest, reference = brecanY_test)   
svmConfMatrix

# Predict probs on the test set
# svmRProbPred <- as.numeric(predict(svmRTune, brecanX_test, type = "prob")$M)  #$posterior
svmRProbPred <- predict(svmRTune, brecanX_test, type = "prob")
svmRProbPred_TgtRes <- svmRProbPred$M #svmRProbPred_TgtRes <- svmRProbPred[, 3]

# Create a ROC object
library("pROC") 
svmRROC <- roc(response = brecanY_test, 
                          predictor = svmRProbPred_TgtRes,
                          levels=rev(levels(brecanY_test)))

print(svmRROC)

# find AUC
svmRAUC <- auc(svmRROC)
print(svmRAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt. 
plot(svmRTune, 
     main= "SVM Model (with radial kernel)", 
     lwd=1.3, col=2)


# multiclass ROC curve
tryCatch({
  plot(x = svmRROC, #$predictor, y = svmRROC$response,
       main = "SVM Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = 4, #c("orange", "darkgreen", "red"), 
       lty = 1:3, lwd= 1.5)
   
  
},
error = function(e) {
  print(paste("Error:", e))
})


# Needs at least 2 tuning parameters with multiple values
plot(svmRTune, plotType = "level",
     main="SVM Model per Tuning Parameters")





## 5. K-Nearest Neighbors model -- KNN
## ************************************************************

#ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)

## tuning param: k components
#tuneGrid = data.frame(.k = c(4*(0:5)+1, 10*(0:5)+1, 20*(1:5)+1, 25*(2:9)+1, 50*(2:9)+1)), ## 21 is the best
knnGrid = data.frame(.k = 1:50)

library(caret)
set.seed(360)
knnTune <- caret::train(x = brecanX_train, 
                        y = brecanY_train,
                        method = "knn",
                        metric = "Kappa",
                        preProc = c("center", "scale"),
                        tuneGrid = knnGrid,
                        trControl = ctrl)

knnTune

#predict on test data
knnPredTest <- predict(knnTune, newdata = brecanX_test)
knnPredTest

postResample(knnPredTest, brecanY_test)

knnConfMatrix <- confusionMatrix(data = knnPredTest, reference = brecanY_test)   
knnConfMatrix

# Predict probs on the test set
# knnProbPred <- as.numeric(predict(knnTune, brecanX_test, type = "prob")$M)  #$posterior
knnProbPred <- predict(knnTune, brecanX_test, type = "prob")
knnProbPred_TgtRes <- knnProbPred$M #knnProbPred_TgtRes <- knnProbPred[, 3]

# Create a ROC object
library("pROC") 
knnROC <- roc(response = brecanY_test, 
                         predictor = knnProbPred_TgtRes,
                         levels=rev(levels(brecanY_test)))

print(knnROC)

# find AUC
knnAUC <- auc(knnROC)
print(knnAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt.  
plot(knnTune, 
     main= "KNN Model", 
     lwd=1.3, col=2)


# Needs at least 2 tuning parameters with multiple values
plot(knnTune, plotType = "level",
     main="KNN Model per Tuning Parameters")


# multiclass ROC curve
tryCatch({
  plot(x = knnROC, #$predictor, y = knnROC$response,
       main = "KNN Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = 5, #c("orange", "darkgreen", "red"), 
       lty = 1:3, lwd=1.5)
   
  
},
error = function(e) {
  print(paste("Error:", e))
})


# Needs at least 2 tuning parameters with multiple values
plot(knnTune, plotType = "level",
     main="KNN Model per Tuning Parameters")





## 6. Naive Bayes model
## ************************************************************

# naiveBayes in the e1071 package and NaiveBayes in the klaR package. Both offer Laplace corrections
# klaR package ~ flexible, uses conditional density estimates

install.packages("klaR")
library(klaR)

## Tuning parameters: fL (Laplace Correction), usekernel (Distribution Type)
# adjust (Bandwidth Adjustment)

# tuning params: no tuning param needed. fL takes care of nzv in cat vars
nbGrid <- data.frame(.fL = 2, 
                     .usekernel = TRUE, 
                     .adjust = TRUE)

library(klaR)
library(caret)
set.seed(370)
nbFit <- caret::train( x = brecanX_train, 
                       y = brecanY_train,
                       method = "nb",
                       metric = "Kappa",
                       # preProc = c("center", "scale"),
                       tuneGrid = nbGrid,
                       trControl = ctrl)

nbFit

#predict on test data
nbPredTest <- predict(nbFit, newdata = brecanX_test)
nbPredTest

postResample(nbPredTest, brecanY_test)

nbConfMatrix <- confusionMatrix(data = nbPredTest, reference = brecanY_test)   
nbConfMatrix

# Predict probs on the test set
# nbProbPred <- as.numeric(predict(nbFit, brecanX_test, type = "prob")$M)  #$posterior
nbProbPred <- predict(nbFit, brecanX_test, type = "prob")
nbProbPred_TgtRes <- nbProbPred$M #nbProbPred_TgtRes <- nbProbPred[, 3]

# Create a ROC object
library("pROC") 
nbROC <- roc(response = brecanY_test, 
                        predictor = nbProbPred_TgtRes,
                        levels=rev(levels(brecanY_test)))

print(nbROC)

# find AUC
nbAUC <- auc(nbROC)
print(nbAUC)


## Plotting...
# ----------------------------------------------------------

# plot model... verify optimal pt. --- no tuning param***
plot(nbFit, 
     main= "Naive Bayes Model", 
     lwd=1.3, col = 2)



# multiclass ROC curve
tryCatch({
  plot(x = nbROC, #$predictor, y = nbROC$response,
       main = "Naive Bayes Model: ROC Curve", 
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)",
       col = 6, #c("orange", "darkgreen", "red"), 
       lty = 1:3, lwd=1.5)
 # legend("topright", legend = levels(brecanY_test), 
 #        col = c("darkorange", "darkgreen", "red"),  
 #        pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
 #        inset = c(0, -0.08), xpd = TRUE)
  
},
error = function(e) {
  print(paste("Error:", e))
})


# Needs at least 2 tuning parameters with multiple values
plot(nbFit, plotType = "level",
     main="Naive Bayes Model per Tuning Parameters")




## ***************************************
## d) For the optimal model for the biological predictors, what are 
## the top five important ## predictors?


impVals=varImp(knnTune)
str(impVals)
impVals$importance

impVals


# top 5. 

plot(impVals, 
     top = 5, 
     scales = list(y = list(cex = .8)),
     col = 14,
     main="Optimal Model: KNN Model with Top 5 Predictors Importance"
)


