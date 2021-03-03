# Titanic Model Building 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('Dataset/titanic_train_after_fe.csv')
test = pd.read_csv('Dataset/titanic_test_after_fe.csv')
result = pd.read_csv('Dataset/gender_submission.csv')
result = result['Survived']

train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)
test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)

target = train['Survived']
features = train.drop(['Survived'], axis = 1)

# Standard Scaler for scaling the features value
sc = StandardScaler()
features = sc.fit_transform(features)
test = sc.transform(test)

# Implementing the Logistic Regression
lg_regressor = LogisticRegression()
lg_regressor.fit(features, target)
lgr_predict = lg_regressor.predict(test)

# It's time to Measure the performance of the Logistic Regression model
lgr_cm = confusion_matrix(result, lgr_predict)
print('Logistic Regression Model\'s Confusion Matrix : {}'.format(lgr_cm))
lgr_accuracy = accuracy_score(result, lgr_predict)
print('Logistic Regression Model\'s Accuracy Score : {:.2f}'.format(lgr_accuracy*100))
lgr_f1 = f1_score(result, lgr_predict)
print('Logistic Regression Model\'s F1 Score : {:.2f}'.format(lgr_f1*100))
lgr_precision = precision_score(result, lgr_predict)
print('Logistic Regression Model\'s Precision Score : {:.2f}'.format(lgr_precision*100))
lgr_recall = recall_score(result, lgr_predict)
print('Logistic Regression Model\'s Recall Score : {:.2f}'.format(lgr_recall*100))

# Implementing the k-Nearest Neighbors
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(features, target)
knn_predict = knn_classifier.predict(test)

# It's time to Measure the performance of the K-Nearest Neigbhors
knn_cm = confusion_matrix(result, knn_predict)
print('K-Nearest Neighbors Model\'s Confusion Matrix : {}'.format(knn_cm))
knn_accuracy = accuracy_score(result, knn_predict)
print('K-Nearest Neighbors Model\'s Accuracy Score : {:.2f}'.format(knn_accuracy*100))
knn_f1 = f1_score(result, knn_predict)
print('K-Nearest Neighbors Model\'s F1 Score : {:.2f}'.format(knn_f1*100))
knn_precision = precision_score(result, knn_predict)
print('K-Nearest Neighbors Model\'s Precision Score : {:.2f}'.format(knn_precision*100))
knn_recall = recall_score(result, knn_predict)
print('K-Nearest Neighbors Model\'s Recall Score : {:.2f}'.format(knn_recall*100))

# Implementing the Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(features, target)
dt_predict = dt_classifier.predict(test)

# It's time to Measure the performance of the Decision Tree
# It's time to Measure the performance of the K-Nearest Neigbhors
dt_cm = confusion_matrix(result, dt_predict)
print('K-Nearest Neighbors Model\'s Confusion Matrix : {}'.format(dt_cm))
dt_accuracy = accuracy_score(result, dt_predict)
print('Decision Tree Model\'s Accuracy Score : {:.2f}'.format(dt_accuracy*100))
dt_f1 = f1_score(result, dt_predict)
print('Decision Tree Model\'s F1 Score : {:.2f}'.format(dt_f1*100))
dt_precision = precision_score(result, dt_predict)
print('Decision Tree Model\'s Precision Score : {:.2f}'.format(dt_precision*100))
dt_recall = recall_score(result, dt_predict)
print('Decision Tree Model\'s Recall Score : {:.2f}'.format(dt_recall*100))

# Implementing the Random Forest Model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(features, target)
rf_predict = rf_classifier.predict(test)

# It's time to Measure the performance of the Random Forest Model
rf_cm = confusion_matrix(result, rf_predict)
print('Random Forest Model\'s Confusion Matrix : {}'.format(rf_cm))
rf_accuracy = accuracy_score(result, rf_predict)
print('Random Forest Model\'s Accuracy Score : {:.2f}'.format(rf_accuracy*100))
rf_f1 = f1_score(result, rf_predict)
print('Random Forest Model\'s F1 Score : {:.2f}'.format(rf_f1*100))
rf_precision = precision_score(result, rf_predict)
print('Random Forest Model\'s Precision Score : {:.2f}'.format(rf_precision*100))
rf_recall = recall_score(result, rf_predict)
print('Random Forest Model\'s Recall Score : {:.2f}'.format(rf_recall*100))

# Implementing the Support Vector Machine Classifier(SVC) Model
svc_classifier = SVC()
svc_classifier.fit(features, target)
svc_predict = svc_classifier.predict(test)

# It's time to Measure the performance of the Support Vector Machine Classifier(SVC) Model
svc_cm = confusion_matrix(result, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s Confusion Matrix : {}'.format(svc_cm))
svc_accuracy = accuracy_score(result, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s Accuracy Score : {:.2f}'.format(svc_accuracy*100))
svc_f1 = f1_score(result, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s F1 Score : {:.2f}'.format(svc_f1*100))
svc_precision = precision_score(result, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s Precision Score : {:.2f}'.format(svc_precision*100))
svc_recall = recall_score(result, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s Recall Score : {:.2f}'.format(svc_recall*100))

# Implementing the Linear Support Vector Classifier(LinearSVC) Model
lsvc_classifier = LinearSVC()
lsvc_classifier.fit(features, target)
lsvc_predict = lsvc_classifier.predict(test)

# It's time to Measure the performance of the Support Vector Machine Classifier(SVC) Model
lsvc_cm = confusion_matrix(result, lsvc_predict)
print('LinearSVC Model\'s Confusion Matrix : {}'.format(lsvc_cm))
lsvc_accuracy = accuracy_score(result, lsvc_predict)
print('LinearSVC Model\'s Accuracy Score : {:.2f}'.format(lsvc_accuracy*100))
lsvc_f1 = f1_score(result, lsvc_predict)
print('LinearSVC Model\'s F1 Score : {:.2f}'.format(lsvc_f1*100))
lsvc_precision = precision_score(result, lsvc_predict)
print('LinearSVC Model\'s Precision Score : {:.2f}'.format(lsvc_precision*100))
lsvc_recall = recall_score(result, lsvc_predict)
print('LinearSVC Model\'s Recall Score : {:.2f}'.format(lsvc_recall*100))

# Implementing the Perceptron Model
p_classifier = Perceptron()
p_classifier.fit(features, target)
p_predict = p_classifier.predict(test)

# It's time to Measure the performance of the Perceptron Model
p_cm = confusion_matrix(result, p_predict)
print('Perceptron Model\'s Confusion Matrix : {}'.format(p_cm))
p_accuracy = accuracy_score(result, p_predict)
print('Perceptron Model\'s Accuracy Score : {:.2f}'.format(p_accuracy*100))
p_f1 = f1_score(result, p_predict)
print('Perceptron Model\'s F1 Score : {:.2f}'.format(p_f1*100))
p_precision = precision_score(result, p_predict)
print('Perceptron Model\'s Precision Score : {:.2f}'.format(p_precision*100))
p_recall = recall_score(result, p_predict)
print('Perceptron Model\'s Recall Score : {:.2f}'.format(p_recall*100))

# Implementing the Gaussion Naive Bayes Model
gnb_classifier = GaussianNB()
gnb_classifier.fit(features, target)
gnb_predict = gnb_classifier.predict(test)

# It's time to Measure the performance of the Perceptron Model
gnb_cm = confusion_matrix(result, gnb_predict)
print('Gaussion Naive Bayes Model\'s Confusion Matrix : {}'.format(gnb_cm))
gnb_accuracy = accuracy_score(result, gnb_predict)
print('Gaussion Naive Bayes Model\'s Accuracy Score : {:.2f}'.format(gnb_accuracy*100))
gnb_f1 = f1_score(result, gnb_predict)
print('Gaussion Naive Bayes Model\'s F1 Score : {:.2f}'.format(gnb_f1*100))
gnb_precision = precision_score(result, gnb_predict)
print('Gaussion Naive Bayes Model\'s Precision Score : {:.2f}'.format(gnb_precision*100))
gnb_recall = recall_score(result, gnb_predict)
print('Gaussion Naive Bayes Model\'s Recall Score : {:.2f}'.format(gnb_recall*100))
