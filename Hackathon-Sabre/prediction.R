#Load the libraries
install.packages('caTools')
library(caTools)
library(ROCR)
library(ggplot2)

#Set the working directory
setwd("/Users/amahen/Documents/Hackathons/Hackathon-Sabre")

#Load the data
churndata <- read.csv("churn_data.csv")

#Check for skeweness
pairs(~transaction_per_year+no_of_transactions+time_in_system+
        discount_ratio+churned,
        data=churndata,
        main = "Scatter plot of data")

#Set seed and split the data
set.seed(20)
split <- sample.split(churndata$churned, SplitRatio = 0.75)

#Get train and test data
train_data <- subset(churndata, split == TRUE)
test_data <- subset(churndata, split == FALSE)

#Logistic regression model
model <- glm (churned ~ transaction_per_year + discount_ratio + no_of_transactions
              + time_in_system, 
              data = churndata )
summary(model)

#Prediction
predict <- predict(model, type = 'response')
churndata <- cbind(churndata, predict)
write.csv(churndata,"final_churn.csv")
summary(predict)

#Confusion matrix
table(churndata$churned, predict > 0.5)

#ROC curve
ROCRpred <- prediction(predict, churndata$churned)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))


