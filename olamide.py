#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from  sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Read CSV data
data = pd.read_csv("train.csv")
#preview data
print(data.head())

#Preview data information
print(data.info())

print(data.describe())

#Check missing values
data.isnull().sum()

# percent of missing "Gender"
print('Percent of missing "Gender" records is %.2f%%' %((data['gender'].isnull().sum()/data.shape[0])*100))

print("Number of people who take loan by gender :")
print(data['gender'].value_counts())
#sns.countplot(x='gender', data=data, palette = 'Set2')


# percent of missing "Married"
print('Percent of missing "Married" records is %.2f%%' %((data['married'].isnull().sum()/data.shape[0])*100))

print("Number of people who take a loan by marital status :")
print(data['married'].value_counts())
#sns.countplot(x='married', data=data, palette = 'Set2')

# percent of missing "Dependents"
print('Percent of missing "Dependents" records is %.2f%%' %((data['dependents'].isnull().sum()/data.shape[0])*100))

print("Number of people who take a loan by dependents :")
print(data['dependents'].value_counts())
#sns.countplot(x='dependents', data=data, palette = 'Set2')

# percent of missing "Self_Employed"
print('Percent of missing "Self_Employed" records is %.2f%%' %((data['self_employed'].isnull().sum()/data.shape[0])*100))

print("Number of people who take a loan that are self employed :")
print(data['self_employed'].value_counts())
#sns.countplot(x='self_employed', data=data, palette = 'Set2')

# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
#
# If "Gender" is missing for a given row, I'll impute with Male (most common answer).
# If "Married" is missing for a given row, I'll impute with yes (most common answer).
# If "Dependents" is missing for a given row, I'll impute with 0 (most common answer).
# If "Self_Employed" is missing for a given row, I'll impute with no (most common answer).
# If "LoanAmount" is missing for a given row, I'll impute with mean of data.
# If "Loan_Amount_Term" is missing for a given row, I'll impute with 360 (most common answer).
# If "Credit_History" is missing for a given row, I'll impute with 1.0 (most common answer).

train_data = data.copy()
train_data['gender'].fillna(train_data['gender'].value_counts().idxmax(), inplace=True)
train_data['married'].fillna(train_data['married'].value_counts().idxmax(), inplace=True)
train_data['dependents'].fillna(train_data['dependents'].value_counts().idxmax(), inplace=True)
train_data['self_employed'].fillna(train_data['self_employed'].value_counts().idxmax(), inplace=True)
train_data["loan_amount"].fillna(train_data["loan_amount"].mean(skipna=True), inplace=True)
train_data['loan_amount_term'].fillna(train_data['loan_amount_term'].value_counts().idxmax(), inplace=True)
train_data['credit_history'].fillna(train_data['credit_history'].value_counts().idxmax(), inplace=True)

#Check missing values
train_data.isnull().sum()
print(train_data)

#Convert some object data type to int64
gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}

train_data['gender'] = train_data['gender'].replace(gender_stat)
train_data['married'] = train_data['married'].replace(yes_no_stat)
train_data['dependents'] = train_data['dependents'].replace(dependents_stat)
train_data['education'] = train_data['education'].replace(education_stat)
train_data['self_employed'] = train_data['self_employed'].replace(yes_no_stat)
train_data['residential_area'] = train_data['residential_area'].replace(property_stat)

#Preview data information
print(data.info())
data.isnull().sum()

#Separate feature and target
x = train_data.iloc[:,1:12]
y = train_data.iloc[:,12]

#make variabel for save the result and to show it
classifier = ('Gradient Boosting','Random Forest','Decision Tree','K-Nearest Neighbor','SVM')
y_pos = np.arange(len(classifier))
score = []

clf = GradientBoostingClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuracy of Gradient Boosting classification is %.2f%%' %(scores.mean()*100))

clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuracy of Random Forest classification is %.2f%%' %(scores.mean()*100))

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuracy of Decision Tree classification is %.2f%%' %(scores.mean()*100))

clf = KNeighborsClassifier()
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuracy of KNN classification is %.2f%%' %(scores.mean()*100))

clf  =  svm.LinearSVC(max_iter=5000)
scores = cross_val_score(clf, x, y,cv=5)
score.append(scores.mean())
print('The accuracy of SVC classification is %.2f%%' %(scores.mean()*100))

state=12
test_size = 0.30
X_train, X_val, y_train, y_val = train_test_split(x, y,
    test_size=test_size, random_state=state)
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))