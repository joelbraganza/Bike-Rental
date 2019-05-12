rm(list=ls())
setwd('C:/JOEL/important-PDFs/edwisor/Project2_BikeRental/BikeRental')

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')


lapply(x, require, character.only = TRUE)
rm(x)

# exploratory data-analysis

df = read.csv('day.csv',header=T)
dim(df) # (731,16)
str(df)

# we can use the 'dteday' variable as categorical 'factor', 
# so we need to keep the day information and convert it along with other categorical 'int' to 'factor' variables.
df$dteday = substr(df$dteday,start=9,stop=10)
df$dteday = factor(df$dteday,labels=(1:length(levels(factor(df$dteday)))))

# categorical and continuous independent variables
cat_names = c('dteday','season','yr','mnth','holiday','weekday','workingday','weathersit')
cnames = c('instant','temp','atemp','hum','windspeed','casual','registered')

df$season = factor(df$season)
df$yr = factor(df$yr)
df$mnth = factor(df$mnth)
df$holiday = factor(df$holiday)
df$weekday = factor(df$weekday)
df$workingday = factor(df$workingday)
df$weathersit = factor(df$weathersit)

# defining function for returning missing value datframe
missing <- function(dataframe) {
  missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
  missing_val$Columns = row.names(missing_val)
  names(missing_val)[1] =  "Missing_percentage"
  missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df)) * 100
  missing_val = missing_val[order(-missing_val$Missing_percentage),]
  row.names(missing_val) = NULL
  missing_val = missing_val[,c(2,1)]
  return(missing_val)
}


missing(df)
# no missing values here



#Doing visualizations(histograms) to see the probability-distribution of the data

hist(df$temp,main = 'histogram of temperature',xlab='temperatures')
abline(v=median(df$temp),col="red") #median
abline(v=mean(df$temp),col="blue") #mean
posStd = (median(df$temp)+sd(df$temp))
negStd = (median(df$temp)-sd(df$temp))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$temp)) # for kernel-density-plot of the data-distribution

hist(df$instant,main = 'histogram of instance',xlab='instances')
abline(v=median(df$instant),col="red") #median
abline(v=mean(df$instant),col="blue") #mean
posStd = (median(df$instant)+sd(df$instant))
negStd = (median(df$instant)-sd(df$instant))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$instant)) # for kernel-density-plot of the data-distribution

hist(df$atemp,main = 'histogram of atemp',xlab='a temperatures')
abline(v=median(df$atemp),col="red") #median
abline(v=mean(df$atemp),col="blue") #mean
posStd = (median(df$atemp)+sd(df$atemp))
negStd = (median(df$atemp)-sd(df$atemp))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$atemp)) # for kernel-density-plot of the data-distribution

hist(df$hum,main = 'histogram of hum',xlab='humidity')
abline(v=median(df$hum),col="red") #median
abline(v=mean(df$hum),col="blue") #mean
posStd = (median(df$hum)+sd(df$hum))
negStd = (median(df$hum)-sd(df$hum))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$hum)) # for kernel-density-plot of the data-distribution

hist(df$windspeed,main = 'histogram of windspeed',xlab='windspeed')
abline(v=median(df$windspeed),col="red") #median
abline(v=mean(df$windspeed),col="blue") #mean
posStd = (median(df$windspeed)+sd(df$windspeed))
negStd = (median(df$windspeed)-sd(df$windspeed))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$windspeed)) # for kernel-density-plot of the data-distribution

hist(df$casual,main = 'histogram of casual',xlab='casual')
abline(v=median(df$casual),col="red") #median
abline(v=mean(df$casual),col="blue") #mean
posStd = (median(df$casual)+sd(df$casual))
negStd = (median(df$casual)-sd(df$casual))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$casual)) # for kernel-density-plot of the data-distribution

hist(df$registered,main = 'histogram of registered',xlab='registered')
abline(v=median(df$registered),col="red") #median
abline(v=mean(df$registered),col="blue") #mean
posStd = (median(df$registered)+sd(df$registered))
negStd = (median(df$registered)-sd(df$registered))
abline(v=posStd,col="purple") #positive std-dev
abline(v=negStd,col="purple") # negative std-dev
plot(density(df$registered)) # for kernel-density-plot of the data-distribution

# from the probability-density-functions, we can conclude that:
#  'instant' variable is more like an index, and it shows uniformity that's why, so we can ignore it's inferences,
#  'temp,'atemp','humidity' and 'windspeed' variables have a pretty-decent normal-distribution as shown in the graph as their-
#   mean and median line-plots are pretty close and nearly-coinciding in some cases, even the kernel-distribution line shows that
#   most of the data is within one standard-deviation both sides of the median
#   However, the 'casual' variable is skewed more towards the right inspite of majority of the data being in a normal-range.
#   Same results we got in the python-analysis

# Outlier-analysis
# ## BoxPlots - Distribution and Outlier Check
boxplot(df$instant)
boxplot(df$temp)
boxplot(df$atemp)
boxplot(df$hum)
boxplot(df$windspeed)
boxplot(df$casual)
boxplot(df$registered)

# from the outlier-analyis, we can confirm that:
# 'hum' variable has some outlier on the lower-side
# 'windspeed' variable also has considerable outliers on the upper-side
# 'casual' variable which showed outliers on the right in the histogram has many outliers on the upper-side
#  so far ( same as PYTHON results)

# Let's work on the outlier-removal
intantOuts = boxplot(df$instant)$out     # vector of outliers in 'instant' variable
tempOuts   = boxplot(df$temp)$out        #                       'temp'
atempOuts  = boxplot(df$atemp)$out       #                       'atemp'
humOuts    = boxplot(df$hum)$out         #                       'hum' ( 2 found)
windspeedOuts = boxplot(df$windspeed)$out#                       'windspeed' (many found)
casualOuts = boxplot(df$casual)$out      #                        'casual'(many found)

# so we need to replace the outliers from 'hum','casual','windspeed' with NA

df[which(df$hum %in%humOuts),"hum"]=NA
df[which(df$casual %in%casualOuts),'casual']=NA
df[which(df$windspeed %in%windspeedOuts),'windspeed']=NA  


missing(df)
# after outlier removal, this is the percentage of missing-data in these variables
# casual       6.0191518
# windspeed    1.7783858
# hum          0.2735978

df = knnImputation(df, k = 3)
# imputed all missing values

boxplot(df$hum)
boxplot(df$windspeed)
boxplot(df$casual)
# from boxplot we can see no outliers in the above variables

write.csv(df, "imputedBikeRental.csv", row.names = F)
## Correlation Plot 
corrgram(df[,cnames], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
# from the correlation plot we can see that 'temp' and 'atemp' variables are highly correlated

# Doing ANOVA test(analysis-of-variance) test
summary(aov(formula=cnt~dteday, data = df)) # pVal: 1>0.05 (not significant)
summary(aov(formula=cnt~season, data = df)) # pVal: <2e-16 ***<0.05 (significant)
summary(aov(formula=cnt~yr, data = df))  # pVal: <2e-16 *** <0.05 (significant)
summary(aov(formula=cnt~mnth, data = df)) # pVal: <2e-16 *** <0.05 (significant)
summary(aov(formula=cnt~holiday, data = df)) # 0.0648 >0.05 (not significant)
summary(aov(formula=cnt~weekday, data = df)) # pVal: 0.583>0.05 (not significant)
summary(aov(formula=cnt~workingday, data = df)) # pVal: 0.0985>0.05 (not significant)
summary(aov(df$cnt~df$weathersit)) # pVal: <2e-16 *** <0.05 (significant)

# unlike Python, here the one-way ANOVA analysis is showing different results as different p-values:
# here season,yr,mnth,weathersit variables are showing same p values which are less than 0.05 so are significant
# however, dteday,holiday,weekday,workingday variables have p values greater than 0.05 so they are not significant.
# we'll even remove'instant' variable as it is simply an index to our data
# so we'll go ahead with removing the insignificant variables, both categorical and continuous.
df = subset(df, select = -c(instant,atemp,dteday,holiday,weekday,workingday))
cnames=c("temp", "hum", "windspeed", "casual", "registered")
#Let's normalize the continuous variables
for(i in cnames)
{
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/(max(df[,i])-min(df[,i]))
}

#executing the ML models over the processed, normalized data 

# using stratified sampling to divide the data into train and test
set.seed(123) # setting the seed for random indices selection, for reproducing those same results(random-indices)
# training=80%, testing=20%
train.index = sample(1:nrow(df), 0.8 * nrow(df)) 
train = df[ train.index,]
test  = df[-train.index,]

MAPE <- function(actual,predicted){
  return(mean(abs((actual-predicted)/actual) * 100))
}

# developing Linear-regression model over the data
LR_model = lm(cnt ~ ., data = df)

#making predictions for training data
pred_LR_train = predict(LR_model, train[,names(test) != "cnt"])

#making predictions for testing data
pred_LR_test = predict(LR_model,test[,names(test) != "cnt"])

# For training data 
print(postResample(pred = pred_LR_train, obs = train[,10]))
#      RMSE       Rsquared       MAE 
#  275.0217566   0.9801361   134.9540495 

# For testing data 
print(postResample(pred = pred_LR_test, obs = test[,10]))
#       RMSE       Rsquared        MAE 
#   159.6458481   0.9934056    96.0730544 

MAPE(actual=test[,10],predicted = pred_LR_test)
# MAPE: 2.287171%

# Decision tree for regression tree
# Developing  Model on training data
DT_model = rpart(cnt ~., data = df, method = "anova")

#Summary of DT model
summary(DT_model)

#write rules into disk
write(capture.output(summary(DT_model)), "regression_DT_Rules.txt")
# contains all rules, helpful to visualize the trees.

#Lets predict for training data
pred_DT_train = predict(DT_model, train[,names(test) != "cnt"])

#Lets predict for training data
pred_DT_test = predict(DT_model,test[,names(test) != "cnt"])


# For training data 
print(postResample(pred = pred_DT_train, obs = train[,10]))
#      RMSE     Rsquared        MAE 
# 505.7036542   0.9327694 394.8150706

# For testing data
print(postResample(pred = pred_DT_test, obs = test[,10]))
#    RMSE    Rsquared         MAE 
#500.1824178   0.9300899 400.3445492 

MAPE(actual = test[,10], predicted = pred_DT_test)
# 11.32135%
 
# making Random Forest Model on training data
RF_model = randomForest(cnt~., data = df)


# making predictions for training data
pred_RF_train = predict(RF_model, train[,names(test) != "cnt"])

# making predictions for testing data
pred_RF_test = predict(RF_model,test[,names(test) != "cnt"])

# calculate errors training data 
print(postResample(pred = pred_RF_train, obs = train[,10]))
#       RMSE    Rsquared         MAE 
# 184.0418605   0.9921101 114.1403065 

# calculate errors testing data
print(postResample(pred = pred_RF_test, obs = test[,10]))
#   RMSE         Rsquared         MAE 
# 143.2646680   0.9943104  93.1832320 
MAPE(actual=test[,10],predicted = pred_RF_test)
# 2.754638%
