import pandas as pd
import numpy as np
import matplotlib as pyplot

# ijmmporting datatsets
Loan_data=pd.read_csv("Loan_train.CSV")
Loan_data.describe()


Loan_final=pd.read_csv("Loan_test.CSV")
Loan_final.describe()

#box plot by Educationa dn credit history
Loan_data.boxplot(column='ApplicantIncome', by=['Education','Credit_History'])

#Histogram
Loan_data['LoanAmount'].hist(bins=50)
temp=pd.crosstab(Loan_data['Credit_History'],Loan_data['Loan_Status'])
temp.plot(kind='bar', color=['red','blue'])

#missing values in training dataset
#Loan_data['Gender'].value_counts()
#Loan_data['Married'].value_counts()
#Loan_data['Self_Employed'].value_counts()
#Loan_data['Credit_History'].value_counts()
Loan_data['Gender'].fillna('Male', inplace=True)
Loan_data['Married'].fillna('Yes', inplace=True)
Loan_data['Self_Employed'].fillna('No', inplace=True)
Loan_data['Credit_History'].fillna(1, inplace=True)
Loan_data['LoanAmount'].fillna(Loan_data['LoanAmount'].mean(), inplace=True)
Loan_data['Loan_Amount_Term'].fillna(Loan_data['Loan_Amount_Term'].mean(), inplace=True)
Loan_data['Dependents'].fillna(Loan_data['Dependents'].mean(), inplace=True)

#missing values in testing dataset
#Loan_final['Gender'].value_counts()
#Loan_final['Self_Employed'].value_counts()
#Loan_final['Credit_History'].value_counts()
Loan_final['Gender'].fillna('Male', inplace=True)
Loan_final['Self_Employed'].fillna('No', inplace=True)
Loan_final['LoanAmount'].fillna(Loan_final['LoanAmount'].mean(), inplace=True)
Loan_final['Loan_Amount_Term'].fillna(Loan_final['Loan_Amount_Term'].mean(), inplace=True)
Loan_final['Dependents'].fillna(0, inplace=True)
Loan_final['Credit_History'].fillna(1, inplace=True)

#CREATING DUMMIES
#creating dummies for train
dummies_gender_train=pd.get_dummies(Loan_data.Gender, columns=['Male', 'Female'])
dummies_married_train=pd.get_dummies(Loan_data.Married, columns=['Marry_yes', 'Marry_no'])
dummies_Education_train=pd.get_dummies(Loan_data.Education, columns=['NotGraduate', 'Graduate'])
dummies_Selfemployed_train=pd.get_dummies(Loan_data.Self_Employed, prefix=['Self_Employed'])
dummies_CreditHistory_train=pd.get_dummies(Loan_data.Credit_History, columns=['CE_1', 'CE_0'])
dummies_ProprtyArea_train=pd.get_dummies(Loan_data.Property_Area, columns=['Urban', 'Semiurban','Rural'])

#creating dummies for test
dummies_gender_test=pd.get_dummies(Loan_final.Gender, columns=['Male', 'Female'])
dummies_married_test=pd.get_dummies(Loan_final.Married, columns=['Marry_yes', 'Marry_no'])
dummies_Education_test=pd.get_dummies(Loan_final.Education, columns=['NotGraduate', 'Graduate'])
dummies_SelfEmployed_test=pd.get_dummies(Loan_final.Self_Employed, prefix=['Self_Employed'])
dummies_CreditHistory_test=pd.get_dummies(Loan_final.Credit_History, columns=['CE_1', 'CE_0'])
dummies_ProprtyArea_test=pd.get_dummies(Loan_final.Property_Area, columns=['Urban', 'Semiurban','Rural'])

#adding Dummy variables to the dataset
Loan_data=Loan_data.join(dummies_gender_train)
Loan_data=Loan_data.join(dummies_married_train)
Loan_data=Loan_data.join(dummies_Education_train)
Loan_data=Loan_data.join(dummies_Selfemployed_train)
Loan_data=Loan_data.join(dummies_CreditHistory_train)
Loan_data=Loan_data.join(dummies_ProprtyArea_train)
Loan_final=Loan_final.join(dummies_gender_test)
Loan_final=Loan_final.join(dummies_married_test)
Loan_final=Loan_final.join(dummies_Education_test)
Loan_final=Loan_final.join(dummies_SelfEmployed_test)
Loan_final=Loan_final.join(dummies_CreditHistory_test)
Loan_final=Loan_final.join(dummies_ProprtyArea_test)

# dropping variables
Loan_data=Loan_data.drop('Gender', axis=1)
Loan_data=Loan_data.drop('Married', axis=1)
Loan_data=Loan_data.drop('Education', axis=1)
Loan_data=Loan_data.drop('Self_Employed', axis=1)
Loan_data=Loan_data.drop('Credit_History', axis=1)
Loan_data=Loan_data.drop('Property_Area', axis=1)
Loan_data=Loan_data.drop('Loan_ID', axis=1)

Loan_final=Loan_final.drop('Gender', axis=1)
Loan_final=Loan_final.drop('Married', axis=1)
Loan_final=Loan_final.drop('Education', axis=1)
Loan_final=Loan_final.drop('Self_Employed', axis=1)
Loan_final=Loan_final.drop('Credit_History', axis=1)
Loan_final=Loan_final.drop('Property_Area', axis=1)
Loan_final=Loan_final.drop('Loan_ID', axis=1)
 
#Splitting dependant and independenat variables
Load_data_Y=Loan_data.iloc[:,5].values
Loan_data_X=Loan_data.drop('Loan_Status', axis=1)

#TEST TRAIN SPLIT 
from sklearn.model_selection import train_test_split
Loan_train_X,Loan_test_X,Loan_train_Y,Loan_test_Y =train_test_split(Loan_data_X,Load_data_Y, test_size=0.18)

#logisitic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(Loan_train_X,Loan_train_Y)

Loan_predict_LR=model.predict(Loan_test_X)

#calculating accuracy
from sklearn import metrics 
accuracy_LR = metrics.accuracy_score(Loan_predict_LR,Loan_test_Y)
print(accuracy_LR)

#confusion matrix 
from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Loan_test_Y, Loan_predict_LR)
print(CM)
#predicting
Load_final_predict=model.predict(Loan_final)
Load_final_predict=np.array(Load_final_predict, dtype=str)
