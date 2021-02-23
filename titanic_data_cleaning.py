import numpy as np
import pandas as pd

train_data = pd.read_csv('Dataset/train.csv')
test_data = pd.read_csv('Dataset/test.csv')
test_survival = pd.read_csv('Dataset/gender_submission.csv')

# Joining the Survival for test dataset
def test_with_survival_data(test_data, test_survival):
    return pd.merge(test_data, test_survival, how = 'outer')

test_all = test_with_survival_data(test_data, test_survival)

# Concatinating all the feature on the one dataset (In one CSV file)
def all_data(train_data, test_all):
    return pd.concat([train_data, test_all], sort = True).reset_index(drop = True)

full_data = all_data(train_data, test_all)
# Droping the unwanted feature
full_data = full_data.drop(['Unnamed: 0'], axis =1)

# Exporting the full data features in one CSV File
full_data.to_csv('Dataset/titanic_complete.csv', index = False)

titanic_complete = pd.read_csv('Dataset/titanic_complete.csv')
