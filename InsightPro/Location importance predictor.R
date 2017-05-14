#Load the libraries
library(rpart)
library(rpart.plot)
library(party)
library(caTools)
library(ROCR)
library(ggplot2)
#Set the working directory
setwd("C:\\Users\\Divya\\Documents\\DIVYA\\Hackathons\\Go Jek")

#Load the data
locdata <- read.csv("location_importance_data.csv")
length=nrow(locdata)
index<-0.7*length
locdata$Location<-as.factor(locdata$Location)
locdata_train <- locdata[1:index,]
locdata_test <- locdata[(index+1):length,]

#grow tree 
fit <- rpart(Customer.Importance ~ gender + Age +
               Number.of.Orders_1 + Value.of.Orders_1 + Weekend.Orders_1 + 
               Number.of.Orders_2 + Value.of.Orders_2 + Weekend.Orders_2 + 
               Number.of.Orders_3 + Value.of.Orders_3 + Weekend.Orders_3 +            
               Offers.Availed + Location, data = locdata_train, method="class")
summary(fit)
#Predict Output 
rpart.plot(fit,tweak=2)

#prediction on new data

predictedfit= as.data.frame(predict(object=fit,newdata=locdata_test))

#Converting probablities for predicted classes to column names
predictedfit$PredClass<-colnames(predictedfit)[max.col(predictedfit,ties.method="first")]

#preparing data for export to csv
locdata_test_output<-cbind(locdata_test,as.numeric(predictedfit$PredClass))
names(locdata_test_output)[names(locdata_test_output) == 'as.numeric(predictedfit$PredClass)'] <- 'PredClass'

write.csv(locdata_test_output,"PredictedCustomerImportance.csv")

#Confusion matrix
table(locdata_test_output$PredClass, locdata_test$Customer.Importance)

#ROC curve
ROCRpred <- prediction(predictedfit, locdata_test$Customer.Importance)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))

