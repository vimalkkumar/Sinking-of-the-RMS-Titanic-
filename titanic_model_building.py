# Titanic Model Building 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

train = pd.read_csv('Dataset/titanic_train_after_fe.csv')
test = pd.read_csv('Dataset/titanic_test_after_fe.csv')
result = pd.read_csv('Dataset/gender_submission.csv')
result = result['Survived']

train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)
test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)

target = train['Survived']
features = train.drop(['Survived'], axis = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)
# test = sc.fit_transform(test)

# Implementing the Logistic Regression
lg_regressor = LogisticRegression()
lg_regressor.fit(features, target)
lgr_predict = lg_regressor.predict(test)

# It's time to Measure the performance of the Logistic Regression model
lgr_accuracy = accuracy_score(result, lgr_predict)
print('Logistic Regression Model\'s Accuracy Score : {:.2f}'.format(lgr_accu*100))
lgr_f1 = f1_score(result, lgr_predict)
print('Logistic Regression Model\'s F1 Score : {:.2f}'.format(lgr_f1*100))
lgr_precision = precision_score(result, lgr_predict)
print('Logistic Regression Model\'s Precision Score : {:.2f}'.format(lgr_precision*100))
lgr_recall = recall_score(result, lgr_predict)
print('Logistic Regression Model\'s Recall Score : {:.2f}'.format(lgr_recall*100))

