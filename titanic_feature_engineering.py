# Feature Engineering for Titanic Dataset(Sinking of the RMS Titanic)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

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
miss_age = train[train['Title'] == 1]["AgeGroup"].mode() #Student or Adult
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
        
        
# # Mapping the each Age value to its respective numerical value
# age_map = {'Baby': 0, 'Child' : 1, 'Teenager': 2, 'Student': 3, 'Young Adult': 4, 'Adult': 5, 'Senior': 6}

# # Mapping the Age numerical value in both dataset(Train, Test)
# train['AgeGroup'] = train['AgeGroup'].map(age_map)
# test['AgeGroup'] = test['AgeGroup'].map(age_map)

# PassengerId - Droping it


