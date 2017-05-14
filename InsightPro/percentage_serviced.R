#Load the libraries
#install.packages('caTools')
#library(caTools)
library(ROCR)
library(ggplot2)

#Set the working directory
setwd("C:/Users/admin/Documents/She-Hack/InsightPro")

#Load the data
serviceddata <- read.csv("percentage_serviced.csv")

#Check for skeweness
pairs(~ Morning_Orders + Noon_Orders + Evening_Orders + Night_Orders +
        Order_Issues + Available_Drivers + Percentage_Serviced,
      data=serviceddata,
      main = "Scatter plot of data")

#Set seed and split the data
set.seed(20)
split <- sample.split(serviceddata$Percentage_Serviced, SplitRatio = 0.75)

#Get train and test data
train_data <- subset(serviceddata, split == TRUE)
test_data <- subset(serviceddata, split == FALSE)

#Logistic regression model
model <- lm (Percentage_Serviced ~ Morning_Orders + Noon_Orders + Evening_Orders + Night_Orders +
               Order_Issues + Available_Drivers,
              data = serviceddata)
summary(model)

#Prediction
predict <- predict(model, type = 'response')
servicedata <- cbind(serviceddata, predict)
write.csv(servicedata,"predicted_percentage_serviced.csv")
summary(predict)
anova(model)
