# Set Working Directory
rm(list=ls())
setwd("F:/Pgda/Telecom_Churn")

# Install and Load the required package
library(MASS)
library(car)
library(e1071)
library(ROCR)
library(caret)
library(ggplot2)
library(class)
library("Hmisc")

# Load the given files.
churn <- read.csv("churn_data.csv", quote="", stringsAsFactors=FALSE)
str(churn)
customer <- read.csv("customer_data.csv", quote="", stringsAsFactors=FALSE)
str(customer)
internet <- read.csv("internet_data.csv", quote="", stringsAsFactors=FALSE)
str(internet)

# Collate the 3 files in a single file.
churn <- merge(x = churn, y = customer, by = "customerID", all = TRUE)
churn <- merge(x= churn, y= internet, by = "customerID", all= TRUE)



# Understand the structure of the collated file.

str(churn)
sum(duplicated(churn))
# No dulicates are found  in the data frame

# Missing value treatment 

sum(is.na(churn));
colnames(churn)[colSums(is.na(churn)) > 0]

#Col TotalCharges is having 11 Missing values 
churn$TotalCharges[is.na(churn$TotalCharges)] <- round(mean(churn$TotalCharges, na.rm = TRUE))

#trimws(churn)
 

# Make bar charts to find interesting relationships between variables.
ggplot(churn, aes(x=gender, fill = Churn)) + geom_bar()
ggplot(churn, aes(x=PaymentMethod, fill = Churn)) + geom_bar()
ggplot(churn, aes(x=Contract, fill = Churn)) + geom_bar()
ggplot(churn, aes(x=tenure, fill = Churn)) + geom_bar(position = 'fill')
ggplot(churn, aes(x=SeniorCitizen, fill = Churn)) + geom_bar(position = 'fill')


# Make Box plots for numeric variables to look for outliers. 

#Box plot for Tenure
boxplot(churn$tenure)
boxplot.stats(churn$tenure)

#Box plot for Monthly charges
boxplot(churn$MonthlyCharges)
boxplot.stats(churn$MonthlyCharges)

#Box plot for Total Charges
boxplot(churn$TotalCharges)
boxplot.stats(churn$TotalCharges)


#Perform Binning of the continous  variables tenure, Monthly and Yearly charges

churn$tenure <- cut(churn$tenure, seq(0,75,25), right=FALSE, labels=c('0-25','25-50','50-75'))   
churn$MonthlyCharges <- cut(churn$MonthlyCharges, seq(0,120,30), right=FALSE, labels=c('0-30','30-60','60-90','90+'))
churn$TotalCharges <- cut(churn$TotalCharges, seq(0,10000,2500), right=FALSE, labels=c('0-2500','2500-5000','5000-7500','7500-10000'))

# Bring the variables in the correct format
# Data transformation
chr.churn <- subset(churn,select = -c(1,11))
factor.churn <- data.frame(sapply(chr.churn, function(x) factor(x)))
num.churn <- subset(churn,select = c(11))
churn_data <- cbind(factor.churn,num.churn)
dummies <- sapply(factor.churn, function(x) data.frame(model.matrix(~x -1, data = factor.churn)) [, -1])
churn <- cbind(dummies,num.churn)


# K-NN Model:

# Bring the data in the correct format to implement K-NN model.
churn$Churn <- as.factor(churn$Churn)

# Implement the K-NN model for optimal K.
set.seed(2)
s=sample(1:nrow(churn),0.7*nrow(churn))

data_train=churn[s,]
data_test=churn[-s,]

cl <- data_train[, 16]

#Training and testing data without the true labels
data_train <- data_train[,-16]
data_test1 <- data_test[, -16]

#KNn with 1NN
impknn1 <- knn(data_train,data_test1, cl, k = 1,
               prob = TRUE)
 
table(impknn1,data_test[,16])
confusionMatrix(impknn1, data_test[,16], positive ="1" )


#KNN - 3 Nearest neighbours
impknn3 <- knn(data_train,data_test1, cl, k = 3,
               prob = TRUE)
table(impknn3,data_test[,16])
confusionMatrix(impknn3, data_test[,16], positive ="1")

#KNN - 5 Nearest Neighbours
impknn5 <- knn(data_train,data_test1, cl, k = 5,
               prob = TRUE)
attr(impknn5,"prob") <- ifelse(impknn5==1,attr(impknn5,"prob"),1 - attr(impknn5,"prob"))
table(impknn5,data_test[,16])
confusionMatrix(impknn5, data_test[,16], positive ="1")

#KNN - 7 Nearest Neighbours
impknn7 <- knn(data_train,data_test1, cl, k = 7,
               prob = TRUE)
table(impknn7,data_test[,16])
confusionMatrix(impknn7, data_test[,16], positive = "1")


#KNN - 9 Nearest Neighbours
impknn9 <- knn(data_train,data_test1, cl, k = 9,
               prob = TRUE)
attr(impknn9,"prob") <- ifelse(impknn9==1,attr(impknn9,"prob"),1 - attr(impknn9,"prob"))
table(impknn9,data_test[,16])
confusionMatrix(impknn9, data_test[,16], positive = "1")



#calculating the values for ROC curve
pred <- prediction(attr(impknn9,"prob"), data_test[,"Churn"])
perf <- performance(pred,"tpr","fpr")

# changing params for the ROC plot - width, etc
par(mar=c(5,5,2,2),xaxs = "i",yaxs = "i",cex.axis=1.3,cex.lab=1.4)

# plotting the ROC curve
plot(perf,col="black",lty=3, lwd=3)

# calculating AUC
auc <- performance(pred,"auc")
unlist(auc@y.values)

#Finding an optimal K
#We will use cross validation to do this.
# Splitting into training and testing
set.seed(2)
s_optk=sample(1:nrow(churn),0.7*nrow(churn))
data_train_optk=churn[s_optk,]
data_test_optk=churn[-s_optk,]


#Using the train() command to find the best K.
model <- train(Churn~., data=data_train_optk,
               method='knn',
               tuneGrid=expand.grid(.k=1:50),
               metric='Accuracy',
               trControl=trainControl(
                 method='repeatedcv', 
                 number=10, 
                 repeats=15))


#Generating the plot of the model
model
plot(model)


# Naive Bayes Model:

# Bring the data in the correct format to implement Naive Bayes algorithm.

churn_data$SeniorCitizen <- factor(churn_data$SeniorCitizen)
str(churn_data)
set.seed(2)
train_indices = sample(1:nrow(churn_data),0.7*nrow(churn_data))
Bayes_train=churn_data[train_indices,]
Bayes_test=churn_data[-train_indices,]

# Implement the Naive Bayes algorithm.
model <- naiveBayes(Churn ~. , data = Bayes_train)
pred <- predict(model, Bayes_test) ## test class without label


table(pred,Bayes_test$Churn)

confusionMatrix(pred,Bayes_test$Churn,positive = "Yes")

#Calculate AUC curve for Naive Bayes

predraw_bayes <- predict(model,Bayes_test,type = "raw")
predprob_bayes <- predraw_bayes[,2]
realvec_bayes <- ifelse(Bayes_test$Churn=="Yes",1,0)
pr_bayes <- prediction(predprob_bayes,realvec_bayes)
perf_bayes <- performance(pr_bayes, "tpr", "fpr")
auc_bayes <- performance(pr_bayes, "auc")
unlist(auc_bayes@y.values)

# plotting the ROC curve
plot(perf_bayes)





# Logistic Regression:

# Bring the data in the correct format to implement Logistic regression model.

# Note: that data has already been brought in correct format above @ # Data transformation   


# Implement the Logistic regression algorithm and use stepwise selection 
library(caTools)
set.seed(100)
split_churn <- sample.split(churn$Churn,SplitRatio = 0.7) 
churn_train <- churn[split_churn,]
churn_test <-  churn[!(split_churn),]

intial_model <- glm(Churn~., data = churn_train, family = "binomial")
summary(intial_model)
best_model <- step(intial_model, direction = "both" )
summary(best_model);
vif(best_model)
# Variables to have VIF less than 2 
# Make the final logistic regression model.
# remove MonthlyCharges.x60.90 
model_1 = glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + PhoneService + 
                Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                PaymentMethod.xCredit.card..automatic. + PaymentMethod.xElectronic.check + 
                Dependents + MultipleLines.xYes + 
                InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xYes + 
                OnlineBackup.xYes + TechSupport.xYes + StreamingMovies.xYes + 
                SeniorCitizen, family = "binomial", data = churn_train)
summary(model_1)
vif(model_1)

# Remove variable dependents
model_2 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + PhoneService + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 PaymentMethod.xCredit.card..automatic. + PaymentMethod.xElectronic.check + 
                 MultipleLines.xYes + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes + OnlineBackup.xYes + 
                 TechSupport.xYes + StreamingMovies.xYes + SeniorCitizen, 
               family = "binomial", data = churn_train)
summary(model_2)
vif(model_2)

# Remove PaymentMethod.xCredit.card..automatic. 

model_3 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + PhoneService + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 + PaymentMethod.xElectronic.check + 
                 MultipleLines.xYes + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes + OnlineBackup.xYes + 
                 TechSupport.xYes + StreamingMovies.xYes + SeniorCitizen, 
               family = "binomial", data = churn_train)

summary(model_3)
vif(model_3)

# Remove OnlineBackup.xYes 
model_4 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + PhoneService + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 + PaymentMethod.xElectronic.check + 
                 MultipleLines.xYes + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes +  
                 TechSupport.xYes + StreamingMovies.xYes + SeniorCitizen, 
               family = "binomial", data = churn_train)
summary(model_4)
vif(model_4)

# Remove PhoneService
model_5 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 + PaymentMethod.xElectronic.check + 
                 MultipleLines.xYes + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes +  
                 TechSupport.xYes + StreamingMovies.xYes + SeniorCitizen, 
               family = "binomial", data = churn_train)
summary(model_5)
vif(model_5)

# Remove MultipleLines.xYes
model_6 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 + PaymentMethod.xElectronic.check + 
                 + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes +  
                 TechSupport.xYes + StreamingMovies.xYes + SeniorCitizen, 
               family = "binomial", data = churn_train)
summary(model_6)
vif(model_6)

# Remove Senior Citizen 
model_7 <- glm(formula = Churn ~ tenure.x25.50 + tenure.x50.75 + 
                 Contract.xOne.year + Contract.xTwo.year + PaperlessBilling + 
                 + PaymentMethod.xElectronic.check + 
                 + InternetService.xFiber.optic + 
                 InternetService.xNo + OnlineSecurity.xYes +  
                 TechSupport.xYes + StreamingMovies.xYes , 
               family = "binomial", data = churn_train)


summary(model_7)
vif(model_7)

# model_7 is the final model with most significant varibales and  having VIF < 2 
model_final <- model_7


# C- Statestic  calculation for train and test data 
churn_train$predicted_prob = predict(model_final, type = "response")
rcorr.cens(churn_train$predicted_prob,churn_train$Churn)

churn_test$predicted_prob = predict(model_final, newdata =churn_test, type = "response") 
rcorr.cens(churn_test$predicted_prob,churn_test$Churn)

# K statestics for train and test
model_score <- prediction(churn_train$predicted_prob,churn_train$Churn)
model_perf <- performance(model_score, "tpr", "fpr")
plot(model_perf)

perf.auc <- performance(model_score, measure = "auc")
unlist(perf.auc@y.values)

ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf,"x.values")[[1]])
ks <- max(ks_table) 
ks
decile_No <- ceiling(which(ks_table == ks)/ nrow(churn_train))
decile_No 

model_score <- prediction(churn_test$predicted_prob,churn_test$Churn)
model_perf <- performance(model_score, "tpr", "fpr")
plot(model_perf)


perf.auc <- performance(model_score, measure = "auc")
unlist(perf.auc@y.values)

ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf,"x.values")[[1]])
ks <- max(ks_table) 
ks
decile_No <- ceiling(which(ks_table == ks)/ nrow(churn_test))
decile_No

# Confusion matrix for test and tarin data with diff threshold value
confusionMatrix(as.numeric(churn_train$predicted_prob > 0.3),churn_train$Churn ,positive = '1')
confusionMatrix(as.numeric(churn_test$predicted_prob > 0.3), churn_test$Churn, positive = '1')

confusionMatrix(as.numeric(churn_train$predicted_prob > 0.5),churn_train$Churn ,positive = '1')
confusionMatrix(as.numeric(churn_test$predicted_prob > 0.5), churn_test$Churn, positive = '1')

confusionMatrix(as.numeric(churn_train$predicted_prob > 0.7),churn_train$Churn ,positive = '1')
confusionMatrix(as.numeric(churn_test$predicted_prob > 0.7), churn_test$Churn, positive = '1')

# Plot of the variables in the final model
ggplot(churn_data, aes(x =churn_data$InternetService, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n InternetService \n", y = "\n Telecom Customer Count \n", title = " \n Churn of Customers : InternetService \n")
ggplot(churn_data, aes(x =churn_data$tenure, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n Tenure \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : Tenure \n");
ggplot(churn_data, aes(x =churn_data$Contract, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n Contract \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : Contract \n");
ggplot(churn_data, aes(x =churn_data$PaymentMethod, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n PaymentMethod \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : PaymentMethod \n");
ggplot(churn_data, aes(x =churn_data$TechSupport, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n TechSupport \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : TechSupport \n");	
ggplot(churn_data, aes(x =churn_data$OnlineSecurity, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n OnlineSecurity \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : OnlineSecurity \n");	
ggplot(churn_data, aes(x =churn_data$StreamingMovies, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n StreamingMovies \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : StreamingMovies \n");
ggplot(churn_data, aes(x =churn_data$PaperlessBilling, fill = factor(churn$Churn))) + geom_bar() + scale_fill_discrete(name ="Churn",labels=c("No","Yes")) + labs( x = "\n PaperlessBilling \n", y = "\n Telecom Customer Count \n", title = "\nChurn of Customers : PaperlessBilling \n");	

# SVM:

# Bring the data in the correct format to implement the SVM algorithm.
set.seed(1)

churn$Churn = factor(churn$Churn)
train.indices = sample(1:nrow(churn),0.7*nrow(churn))
train.data = churn[train.indices,]
test.data = churn[-train.indices,]
# Implement the SVM algorithm using the optimal cost.

model.svm.0 = svm(Churn~.,data=train.data,kernel="linear",probability=TRUE,cost = 0.01)
summary(model.svm.0)

model.svm.1 = svm(Churn~.,data=train.data,kernel="linear",probability=TRUE,cost = 0.1)
summary(model.svm.1)

model.svm.2 = svm(Churn~.,data=train.data,kernel="linear",probability=TRUE,cost = 0.5)
summary(model.svm.2)


tune.svm  = tune(svm,Churn~.,data=train.data,kernel= "linear",ranges = list(cost= c(0.01,0.01,0.1,0.5,1,10,100)))
summary(tune.svm)
best.mod_Linear = tune.svm$best.model
best.mod_Linear


#best cost = .01 for linear kernel having maximum accuracy and sensitivity
#confusion Matrix calculation

ypred_SVM = predict(model.svm.0,test.data)

table(predicted = ypred_SVM,truth= test.data$Churn)
confusionMatrix(ypred_SVM,test.data$Churn,positive = "1")

# plotting the ROC curve
predraw_SVM <- predict(model.svm.0,test.data,probability=TRUE, decision.values=TRUE)
predprob_SVM <- attr(predraw_SVM, "probabilities")[, 2]

pr_SVM <- prediction(predprob_SVM,test.data$Churn)
perf <- performance(pr_SVM, "tpr", "fpr")

plot(perf)


# calculating AUC
auc_SVM <- performance(pr_SVM, "auc")
unlist(auc_SVM@y.values)


