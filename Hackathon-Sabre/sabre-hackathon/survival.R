#Load the libraries
library(survival)

#Set the working directory
setwd("/Users/amahen/Documents/Hackathons/Hackathon-Sabre")

#Load the data
churndata <- read.csv("churn_data.csv")
attach(churndata)


#Create a "survival object" for each observation, using time and churn data.
churndata$survival <- Surv(churndata$time_in_system,churndata$churned == 1)

#Fit a basic survival curve using the data
fit <- survfit(survival ~ 1, data = churndata)

#Plot the survival curve
plot(fit, lty = 1, mark.time = FALSE, ylim=c(.75,1), xlab = 'Time in System', ylab = 'Percent Surviving')
title(main = 'Customer Survival Curve')


#Semi-parametric solution
system_time <- churndata$time_in_system
ratio_discount <- churndata$discount_ratio
no_transactions <- churndata$no_of_transactions
value_transaction <- churndata$transaction_per_year
X <- cbind(system_time,ratio_discount,no_transactions,value_transaction)

#Fit a cox regression model           
coxph <- coxph(Surv(churndata$time_in_system,churndata$churned)~X,method = "breslow")
summary(coxph)

