##### dataset

dim(trainRows)

dim(brecanX_train)
length(brecanY_train)
dim(brecanX_test)
length(brecanY_test)




#### BUILDING MODELS... ## Linear Classification Models 

set.seed(400) 
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                    # summaryFunction = twoClassSummary, #defaultSummary
                     classProbs = TRUE, 
                     savePredictions = TRUE)

### LGOCV- repeated training/test splits (25 reps, 75%) ## Leave Group Out cross-validation
#ctrl <- trainControl(method = "LGOCV",
#                     summaryFunction = twoClassSummary,
#                     classProbs = TRUE,
#                     ##index = list(simulatedTest[,1:4]),
#                     savePredictions = TRUE)




############ 1. Logistic Regression ###############

install.packages("MLmetrics")
library(MLmetrics)
library(caret)

#tuning parameter: ?? ...none

set.seed(410)
#lrFit <- caret::train(brecanX_train, brecanY_train,
#                      method = "multinom", metric = "ROC",
#                      #preProcess = c("center","scale"),
#                      trControl = ctrl,
#                      trace = FALSE)

lrFit <- caret::train(brecanX_train, brecanY_train, 
                      method = "glm",
                      metric = "Kappa",
                      trControl = ctrl)

lrFit

## The predict fxn... no need to manually specify the shrinkage amount
#predict on test data
lrPredTest <- predict(lrFit, newdata = brecanX_test)
lrPredTest #sum((lrPredTest == hepabrecanY_test)==TRUE)/length(hepabrecanY_test)

postResample(lrPredTest, brecanY_test)

lrConfMatrix <- confusionMatrix(data = lrPredTest, reference = brecanY_test)   
lrConfMatrix

# Predict probs on the test set
lrProbPred <- predict(lrFit, brecanX_test, type = "prob")
lrProbPred_TgtRes <- lrProbPred$M #lrProbPred_TgtRes <- lrProbPred[, 2]

# Create a ROC object
library("pROC") ##lrROC <- multiclass.roc(response = brecanY_test, predictor = lrProbPred_TgtRes, levels=rev(levels(brecanY_test)))
lrROC <- roc(response = brecanY_test, predictor = lrProbPred_TgtRes, levels=rev(levels(brecanY_test)))
print(lrROC)

# find AUC
lrAUC <- auc(lrROC)
print(lrAUC)

## Plotting ----------------------------------------------------------
# tuning plot... no tuning
plot(lrFit, legacy.axes = TRUE, col=14, lwd=1.2, 
     main="Logistic Regression Model") 

# tuning heatmap
plot(lrFit, plotType = "level", legacy.axes = TRUE, 
     main="lr Model: Tuning Heat-map")

#ROC curve 
par(mfrow = c(1,1)) 
plot(x = lrROC, #$predictor, y = lrROC$response,
     main = "Logistic Regression Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col = "darkorange", # c("darkorange", "darkgreen", "darkred"), 
     lty = 1:3, lwd=1.5)


#legend("topright", legend = levels(brecanY_test), \
#       col = c("darkorange", "darkgreen", "darkred"),  
#       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
#       inset = c(0, -0.08), xpd = TRUE)





############ 2. Linear Discriminant Analysis #############


# install.packages("MLmetrics")
library(MLmetrics)
library(caret)

# tuning params: none

set.seed(420)
ldaFit <- caret::train(brecanX_train, brecanY_train,
                method = "lda", metric = "Kappa",
                #preProcess = c("center","scale"),
                trControl = ctrl,
                trace = FALSE)

ldaFit 

#predict on test data
ldaPredTest <- predict(ldaFit, newdata = brecanX_test)
ldaPredTest #sum((ldaPredTest == hepabrecanY_test)==TRUE)/length(hepabrecanY_test)

postResample(ldaPredTest, brecanY_test)

ldaConfMatrix <- confusionMatrix(data = ldaPredTest, reference = brecanY_test)   
ldaConfMatrix

# Predict probs on the test set
ldaProbPred <- predict(ldaFit, brecanX_test, type = "prob")
ldaProbPred_TgtRes <- ldaProbPred$M #ldaProbPred_TgtRes <- ldaProbPred[, 2]

# Create a ROC object
library("pROC") ##ldaROC <- multiclass.roc(response = brecanY_test, predictor = ldaProbPred_TgtRes, levels=rev(levels(brecanY_test)))
ldaROC <- roc(response = brecanY_test, predictor = ldaProbPred_TgtRes, levels=rev(levels(brecanY_test)))
print(ldaROC)

# find AUC
ldaAUC <- auc(ldaROC)
print(ldaAUC)

## Plotting ----------------------------------------------------------
# tuning plot... no tuning ****
plot(ldaFit, legacy.axes = TRUE, col=14, lwd=1.3, 
     main="Linear Discriminant Analysis Model") 

# tuning heatmap
plot(ldaFit, plotType = "level", legacy.axes = TRUE, main="Linear Discriminant Analysis Model: Tuning Heat-map")

#ROC curve
plot(x = ldaROC, #$predictor, y = ldaROC$response,
     main = "Linear Discriminant Analysis Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col = 4, # c("violet", "darkgreen", "darkred"), 
     lty = 1:3, lwd=1.5)

#legend("topright", legend = levels(brecanY_test), 
#       col = c("darkorange", "darkgreen", "darkred"),  
#       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
#       inset = c(0, -0.08), xpd = TRUE)





############ 3. PLS Discriminant Analysis (PLSDA) #############

install.packages(c("MLmetrics", "glmnet", "pamr", "rms", "sparseLDA", "subselect", "MASS", "pls", "pROC"))

library(pls)
library(MLmetrics)
library(caret)


# tuning param: components retained... 
plsdaGrid <- expand.grid(.ncomp = 1:18)

#set.seed(400) 
#ppctrl <- trainControl(summaryFunction = twoClassSummary,
#                     classProbs = TRUE)

#ppctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
#                     #summaryFunction = twoClassSummary, #defaultSummary
#                     classProbs = TRUE, 
#                     savePredictions = TRUE)

set.seed(430) 
plsdaTune <- caret::train(brecanX_train, brecanY_train,
                         method = "pls", metric = "Kappa",
                         preProcess = c("center","scale"),
                         tuneGrid = plsdaGrid,
                         trControl = ctrl,
                         trace = FALSE)

plsdaTune 

#predict on test data
plsdaPredTest <- predict(plsdaTune, newdata = brecanX_test)
plsdaPredTest #sum((plsdaPredTest == hepabrecanY_test)==TRUE)/length(hepabrecanY_test)

postResample(plsdaPredTest, brecanY_test)

plsdaConfMatrix <- confusionMatrix(data = plsdaPredTest, reference = brecanY_test)   
plsdaConfMatrix

# Predict probs on the test set
plsdaProbPred <- predict(plsdaTune, brecanX_test, type = "prob")
plsdaProbPred_TgtRes <- plsdaProbPred$M #plsdaProbPred_TgtRes <- plsdaProbPred[, 3]

# Create a ROC object
library("pROC") ##plsdaROC <- multiclass.roc(response = brecanY_test, predictor = plsdaProbPred_TgtRes, levels=rev(levels(brecanY_test)))
plsdaROC <- roc(response = brecanY_test, predictor = plsdaProbPred_TgtRes, levels=rev(levels(brecanY_test)))
print(plsdaROC)

# find AUC
plsdaAUC <- auc(plsdaROC)
print(plsdaAUC)

## Plotting ----------------------------------------------------------
# tuning plot
plot(plsdaTune, legacy.axes = TRUE, col=14, lwd=1.3, 
     main="PLS Discriminant Analysis Model") 

# tuning heatmap
plot(plsdaTune, plotType = "level", legacy.axes = TRUE, 
     main="PLS Discriminant Analysis Model: Tuning Heat-map")

#ROC curve
plot(x = plsdaROC, #$predictor, y = plsdaROC$response,
     main = "plsda Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col = 3, #c("darkorange", "darkgreen", "darkred"), 
     lty = 1:3, lwd=1.5)

#legend("topright", legend = levels(brecanY_test), 
#       col = c("darkorange", "darkgreen", "darkred"),  
#       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
#       inset = c(0, -0.08), xpd = TRUE)




############ 4. Penalized Models ##########

## The family argument is related to the distribution of the outcome
## For two classes, use family="binomial" corresponds to logistic regression, 
## For three or more classes, use family="multinomial" is appropriate.
## glmnet defaults this parameter to alpha = 1, corresponding to a
## complete lasso penalty.

library(glmnet)
library(MLmetrics)
library(caret)


# tuning param:
#glmnGrid <- expand.grid(.alpha = seq(0, 1, by=0.1), # c(0, .1, .2, .4, .6, .8, 1),
#                        .lambda = seq(.01, .2, length = 30))

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .3, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

set.seed(440)
glmnTune <- caret::train(brecanX_train, brecanY_train,
                         method = "glmnet", metric = "Kappa",
                         preProcess = c("center","scale"),
                         tuneGrid = glmnGrid,
                         trControl = ctrl,
                         trace = FALSE)

glmnTune

#predict on test data
glmnPredTest <- predict(glmnTune, newdata = brecanX_test)
glmnPredTest #sum((glmnPredTest == hepabrecanY_test)==TRUE)/length(hepabrecanY_test)

postResample(glmnPredTest, brecanY_test)

glmnConfMatrix <- confusionMatrix(data = glmnPredTest, reference = brecanY_test)   
glmnConfMatrix

# Predict probs on the test set
glmnProbPred <- predict(glmnTune, brecanX_test, type = "prob")
glmnProbPred_TgtRes <- glmnProbPred$M #glmnProbPred_TgtRes <- glmnProbPred[, 3]

# Create a ROC object
library("pROC") ##glmnROC <- multiclass.roc(response = brecanY_test, predictor = glmnProbPred_TgtRes, levels=rev(levels(brecanY_test)))
glmnROC <- roc(response = brecanY_test, predictor = glmnProbPred_TgtRes, levels=rev(levels(brecanY_test)))
print(glmnROC)

# find AUC
glmnAUC <- auc(glmnROC)
print(glmnAUC)

## Plotting ----------------------------------------------------------
# tuning plot
plot(glmnTune, legacy.axes = TRUE, lwd=1.3, 
     main="Penalized Model") 

# tuning heatmap
plot(glmnTune, plotType = "level", legacy.axes = TRUE, main="Penalized Model: Tuning Heat-map")

#ROC curve
plot(x = glmnROC, #$predictor, y = glmnROC$response,
     main = "Penalized Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col =  2, #"darkred", #c("darkorange", "darkgreen", "darkred"), 
     lty = 1:3, lwd=1.5)

#legend("topright", legend = levels(brecanY_test), 
#       col = c("darkorange", "darkgreen", "darkred"),  
#       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
#       inset = c(0, -0.08), xpd = TRUE)




########### Nearest Shrunken Centroids ###########
library(caret)

# tuning param: threshold...
nscGrid <- data.frame(.threshold = seq(0, 4, by=0.1))

set.seed(470)
nscTune <- caret::train(x = brecanX_train, y = brecanY_train,
                        method = "pam", metric = "Kappa", 
                        tuneGrid = nscGrid,
                        preProc = c("center", "scale"),
                        trControl = ctrl)
nscTune
 
#predict on test data... no need to manually specify the shrinkage amount
nscPredTest <- predict(nscTune, newdata = brecanX_test)
nscPredTest #sum((nscPredTest == hepabrecanY_test)==TRUE)/length(hepabrecanY_test)

postResample(nscPredTest, brecanY_test)

nscConfMatrix <- confusionMatrix(data = nscPredTest, reference = brecanY_test)   
nscConfMatrix

# Predict probs on the test set
nscProbPred <- predict(nscTune, brecanX_test, type = "prob")
nscProbPred_TgtRes <- nscProbPred$M #nscProbPred_TgtRes <- nscProbPred[, 3]

# Create a ROC object
library("pROC") ##nscROC <- multiclass.roc(response = brecanY_test, predictor = nscProbPred_TgtRes, levels=rev(levels(brecanY_test)))
nscROC <- roc(response = brecanY_test, predictor = nscProbPred_TgtRes, levels=rev(levels(brecanY_test)))
print(nscROC)

# find AUC
nscAUC <- auc(nscROC)
print(nscAUC)

## Plotting ----------------------------------------------------------
# tuning plot
plot(nscTune, legacy.axes = TRUE, col=14, lwd=1.3, 
     main="Nearest Shrunken Centroids Model") 

# tuning heatmap
plot(nscTune, plotType = "level", legacy.axes = TRUE, 
     main="Nearest Shrunken Centroids Model: Tuning Heat-map")

#ROC curve
plot(x = nscROC, #$predictor, y = nscROC$response,
     main = "Nearest Shrunken Centroids Model: ROC Curve", 
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)",
     col = 5, #c("darkorange", "darkgreen", "darkred"), 
     lty = 1:3, lwd=1.5)

#legend("topright", legend = levels(brecanY_test), 
#       col = c("darkorange", "darkgreen", "darkred"),  
#       pch=15, horiz=TRUE, bty='n', cex=0.7, lwd = 1, 
#       inset = c(0, -0.08), xpd = TRUE)




## The predictors function will list the predictors used in the prediction equation
predictors(nscTune)

## variable importance based on the distance between the class centroid and the overall centroid:
varImp(nscTune, scale = FALSE)




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


