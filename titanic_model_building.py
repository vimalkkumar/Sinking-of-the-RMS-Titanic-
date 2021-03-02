# Titanic Model Building 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn 

train = pd.read_csv('Dataset/titanic_train_after_fe.csv')
test = pd.read_csv('Dataset/titanic_test_after_fe.csv')

train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)
test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)

target = train['Survived']
features = train.drop(['Survived'], axis = 1)

