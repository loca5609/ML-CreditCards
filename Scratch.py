import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

dat = pd.read_csv('default_cc_train.csv')



sns.countplot("default.payment.next.month", data = dat, hue = "SEX",palette="Set3")
plt.title("Defaulted by Sex")
plt.xlabel("Defaulted");


sns.violinplot("SEX","AGE", data = dat,palette="Set3")
plt.title("Violin Plot of Age range for each Sex");

sns.violinplot("default.payment.next.month","AGE", data = dat,palette="Set2")
plt.title("Violin Plot of Age range by Default");


violin = sns.violinplot(x="MARRIAGE",y="default.payment.next.month", hue="SEX", data=dat)


# Preprocesing

#Getting rid of 0's in Marriage
dat = dat[dat.MARRIAGE != 0]

#getting rid of 0,5,6 in Education
for i in [0,5,6]:
    dat.EDUCATION.replace(i,4)
 
    
datCat = pd.DataFrame(dat[["SEX","EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5"]])
datNum = pd.DataFrame(dat[["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1","PAY_AMT2","PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"]])

catName = ["SEX","EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5"]

numName= ["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1","PAY_AMT2","PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"]


datUse = dat[["SEX","EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5","LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1","PAY_AMT2","PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"]]

    
    
#Scale data 
class DataSelect(BaseEstimator,TransformerMixin):
    def __init__(self, colNames):
        self.colNames = colNames
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return  X[self.colNames]


#Getting rid of NA's for numerical data
pipeNumeric = Pipeline([
 ("select_cols", DataSelect(numName)),
 ("z-scaling", StandardScaler())
 ])
pipNum = pipeNumeric.fit_transform(datNum)

#Need to deal with strings in categorical before putting in piepline so use get_dummies

class DumCat(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Adding dummies when needed"""
    def fit(self, X,y=None):
        self.dummyDat = pd.get_dummies(X) #Creating dataFrame to run the function
        return self
    def transform(self,X):
        return self.dummyDat #return the dummyDat that has the new columns


#to access, need to Class.fit_transform(dataWantChanged)


#Getting rid of NA's for categorical data
pipeCategory = Pipeline([
 ("select_cols", DataSelect(catName)),
 ("get_dummies", DumCat()),
 ])
pipCat = pipeCategory.fit_transform(datCat)


#Now that the data is processed, need to join the Numerical and Categorical 

full_pipeline = FeatureUnion(transformer_list=[
         ('cat_data',pipeCategory),
        ('num_data',pipeNumeric)
        ])
united = full_pipeline.fit_transform(datUse)
united


# SVM w/o scaling
X = dat.loc[:, dat.columns != "default.payment.next.month"]
y  = dat.iloc[:,24]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)


svm = SVC()

# Training and predictions SVC
svm.fit(X_train,y_train)
predicts = svm.predict(X_test)
accuracy = accuracy_score(y_test,predicts)
print("The accuracy of our Support Vector Classifier is: " + str(accuracy))
#.78105

#Scaled Data SVM   
X = united
y  = dat.iloc[:,24]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)


svm = SVC()

# Training and predictions SVC
svm.fit(X_train,y_train)
predicts = svm.predict(X_test)
accuracy = accuracy_score(y_test,predicts)
print("The accuracy of our Support Vector Classifier is: " + str(accuracy))
#.81894


# =============================================================================
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# X2: Gender (1 = male; 2 = female).
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# X4: Marital status (1 = married; 2 = single; 3 = others).
# X5: Age (year).
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005 
# =============================================================================

# Raw Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_predicts = log_clf.predict(X_test)

log_acc = accuracy_score(log_predicts,y_test)
print("The accuracy of Logistic Regression is: ", log_acc)
# ~77.8% (bad)

rf_clf = RandomForestClassifier(n_estimators=50,random_state=1)
rf_clf.fit(X_train, y_train)
rf_predicts = rf_clf.predict(X_test)

rf_acc = accuracy_score(rf_predicts,y_test)
print("The accuracy of Random Forest Classification is: ", rf_acc)
# ~81.2%

grad_clf = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,max_depth=2,loss="deviance")
grad_clf.fit(X_train,y_train)
grad_predicts = grad_clf.predict(X_test)

grad_acc = accuracy_score(grad_predicts,y_test)
grad_mse = mean_squared_error(grad_predicts,y_test)

print("The MSE for Gradient Boosting is : ", grad_mse)
print("The accuracy of Gradient Boosted Classification is: ", grad_acc)
# ~82 %s
