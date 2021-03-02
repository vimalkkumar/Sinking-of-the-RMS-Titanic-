# Feature Engineering for Titanic Dataset(Sinking of the RMS Titanic)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')

# Extracting the title from the Passengers Name
combine = [train, test]
for title_df in combine:
    title_df['Title'] = title_df.Name.str.extract('([A-Za-z]+)\.', expand = False)

pd.crosstab(train['Title'], train['Sex'])
pd.crosstab(test['Title'], test['Sex'])

# Replacong the various Title with the most common names
for title_df in combine:
    title_df['Title'] = title_df['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    title_df['Title'] = title_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    title_df['Title'] = title_df['Title'].replace('Mlle', 'Miss')
    title_df['Title'] = title_df['Title'].replace('Ms', 'Miss')
    title_df['Title'] = title_df['Title'].replace('Mme', 'Mrs')


pd.crosstab(train['Title'], train['Sex'])
pd.crosstab(test['Title'], test['Sex'])

train[['Title', 'Survived']].groupby('Title').sum()
train[['Title', 'Survived']].groupby('Title').mean()

# Mapping the title with its ordinal Value
title_map = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4, 'Royal': 5}
for title_df in combine:
    title_df['Title'] = title_df['Title'].map(title_map)
    title_df['Title'] = title_df['Title'].fillna(0)

# Labling the AgeGroup based on Age
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

# Filling the Missing Age using Median Age for each age group
master_age = train[train['Title'] == 0]["AgeGroup"].mode() #Baby
miss_age = train[train['Title'] == 1]["AgeGroup"].mode() #Student 
mr_age = train[train['Title'] == 2]["AgeGroup"].mode() #Adult
mrs_age = train[train['Title'] == 3]["AgeGroup"].mode() # Young Adult
rage_age = train[train['Title'] == 4]["AgeGroup"].mode() #Adult
royal_age = train[train['Title'] == 5]["AgeGroup"].mode() #Adult

age_title_map = {0: 'Baby', 1: 'Student', 2: 'Adult', 3: 'Young Adult', 4: 'Adult', 5: 'Adult'}

for i in range(len(train.AgeGroup)):
    if train.AgeGroup[i] == 'Unknown':
        train.AgeGroup[i] = age_title_map[train.Title[i]]
    
for i in range(len(test.AgeGroup)):
    if test.AgeGroup[i] == 'Unknown':
        test.AgeGroup[i] = age_title_map[test.Title[i]]
        
        
# Mapping the each Age value to its respective numerical value
age_map = {'Baby': 0, 'Child' : 1, 'Teenager': 2, 'Student': 3, 'Young Adult': 4, 'Adult': 5, 'Senior': 6}

# Mapping the Age numerical value in both dataset(Train, Test)
train['AgeGroup'] = train['AgeGroup'].map(age_map)
test['AgeGroup'] = test['AgeGroup'].map(age_map)

# Handling Cabin and Creating new feature named Deck 
# Keeping all the first letters of Cabin in a new variable and using 'M' for each missing
train['Deck'] = train['Cabin'].apply(lambda m: m[0] if pd.notnull(m) else 'M')
test['Deck'] = test['Cabin'].apply(lambda m: m[0] if pd.notnull(m) else 'M')

# Grouping the Deck
train['Deck'] = train['Deck'].replace(['A', 'B', 'C', 'T'], 'ABC')
train['Deck'] = train['Deck'].replace(['D', 'E'], 'DE')
train['Deck'] = train['Deck'].replace(['F', 'G'], 'FG')

test['Deck'] = test['Deck'].replace(['A', 'B', 'C', 'T'], 'ABC')
test['Deck'] = test['Deck'].replace(['D', 'E'], 'DE')
test['Deck'] = test['Deck'].replace(['F', 'G'], 'FG')

#Handling Embarked Missing Value
train[train['Embarked'].isnull()]
test[test['Embarked'].isnull()]

# Checking for Passengers who were in Pclass 1, on Deck ABC and paid 80 or less for the Tickets
train.loc[(train['Pclass'] == 1) & (train['Fare'] <= 80) & (train['Deck'] == 'ABC')]['Embarked'].value_counts() 

# Adding the S in missing Embarked
train.loc[train['Embarked'].isnull(), 'Embarked'] = 'S'

# Handling Fare 
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by = 'FareBand', ascending = True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]

# Label Encoding and One hot Encoding
non_numeric_features = ['Embarked', 'Sex', 'Title', 'AgeGroup', 'Fare', 'Deck']

for feature in non_numeric_features:
    train[feature] = LabelEncoder().fit_transform(train[feature])
    test[feature] = LabelEncoder().fit_transform(test[feature])

cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Deck', 'AgeGroup', 'Fare']

encoded_features = []
    
def ohe(dataset):
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(dataset[feature].values.reshape(-1,1)).toarray()
        n = dataset[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n+1)]
        encoded_df = pd.DataFrame(encoded_feat, columns = cols)
        encoded_df.index = dataset.index
        encoded_features.append(encoded_df)
        
# For Train Dataset
ohe(train)    
train = pd.concat([train, *encoded_features], axis = 1)

encoded_features = []
# For Test dataset
ohe(test)    
test = pd.concat([test, *encoded_features], axis = 1)

# Exporting the Train and Test dataset features after feature Engineering in one CSV File
train.to_csv('Dataset/titanic_train_after_fe.csv', index = False)
test.to_csv('Dataset/titanic_test_after_fe.csv', index = False)

titanic_train = pd.read_csv('Dataset/titanic_train_after_fe.csv')
titanic_test = pd.read_csv('Dataset/titanic_test_after_fe.csv')