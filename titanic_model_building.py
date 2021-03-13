# Titanic Model Building 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('Dataset/titanic_train_after_fe.csv')
test = pd.read_csv('Dataset/titanic_test_after_fe.csv')
test_target = pd.read_csv('Dataset/gender_submission.csv')
test_target = test_target['Survived']

# train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)
# test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup', 'Deck'], axis = 1)

train = train[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'AgeGroup', 'Deck']]
test_features = test[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'AgeGroup', 'Deck']]

train_target = train['Survived']
train_features = train.drop(['Survived'], axis = 1)

# # Standard Scaler for scaling the features value
# sc = StandardScaler()
# train_features = sc.fit_transform(train_features)
# test_features = sc.transform(test_features)

# Implementing the Logistic Regression
lg_regressor = LogisticRegression()
lg_regressor.fit(train_features, train_target)
lgr_predict = lg_regressor.predict(test_features)

# It's time to Measure the performance of the Logistic Regression model
lgr_cm = confusion_matrix(test_target, lgr_predict)
print('Logistic Regression Model\'s Confusion Matrix : {}'.format(lgr_cm))
lgr_accuracy = accuracy_score(test_target, lgr_predict) * 100
print('Logistic Regression Model\'s Accuracy Score : {:.2f}'.format(lgr_accuracy))
lgr_f1 = f1_score(test_target, lgr_predict)*100
print('Logistic Regression Model\'s F1 Score : {:.2f}'.format(lgr_f1))
lgr_precision = precision_score(test_target, lgr_predict)*100
print('Logistic Regression Model\'s Precision Score : {:.2f}'.format(lgr_precision))
lgr_recall = recall_score(test_target, lgr_predict)*100
print('Logistic Regression Model\'s Recall Score : {:.2f}'.format(lgr_recall))

# Implementing the k-Nearest Neighbors
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(train_features, train_target)
knn_predict = knn_classifier.predict(test_features)

# It's time to Measure the performance of the K-Nearest Neigbhors
knn_cm = confusion_matrix(test_target, knn_predict)
print('K-Nearest Neighbors Model\'s Confusion Matrix : {}'.format(knn_cm))
knn_accuracy = accuracy_score(test_target, knn_predict)*100
print('K-Nearest Neighbors Model\'s Accuracy Score : {:.2f}'.format(knn_accuracy))
knn_f1 = f1_score(test_target, knn_predict)*100
print('K-Nearest Neighbors Model\'s F1 Score : {:.2f}'.format(knn_f1))
knn_precision = precision_score(test_target, knn_predict)*100
print('K-Nearest Neighbors Model\'s Precision Score : {:.2f}'.format(knn_precision))
knn_recall = recall_score(test_target, knn_predict)*100
print('K-Nearest Neighbors Model\'s Recall Score : {:.2f}'.format(knn_recall))

# Implementing the Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(train_features, train_target)
dt_predict = dt_classifier.predict(test_features)

# It's time to Measure the performance of the Decision Tree
dt_cm = confusion_matrix(test_target, dt_predict)
print('K-Nearest Neighbors Model\'s Confusion Matrix : {}'.format(dt_cm))
dt_accuracy = accuracy_score(test_target, dt_predict)*100
print('Decision Tree Model\'s Accuracy Score : {:.2f}'.format(dt_accuracy))
dt_f1 = f1_score(test_target, dt_predict)*100
print('Decision Tree Model\'s F1 Score : {:.2f}'.format(dt_f1))
dt_precision = precision_score(test_target, dt_predict)*100
print('Decision Tree Model\'s Precision Score : {:.2f}'.format(dt_precision))
dt_recall = recall_score(test_target, dt_predict)*100
print('Decision Tree Model\'s Recall Score : {:.2f}'.format(dt_recall))

# Implementing the Random Forest Model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_features, train_target)
rf_predict = rf_classifier.predict(test_features)

# It's time to Measure the performance of the Random Forest Model
rf_cm = confusion_matrix(test_target, rf_predict)
print('Random Forest Model\'s Confusion Matrix : {}'.format(rf_cm))
rf_accuracy = accuracy_score(test_target, rf_predict)*100
print('Random Forest Model\'s Accuracy Score : {:.2f}'.format(rf_accuracy))
rf_f1 = f1_score(test_target, rf_predict)*100
print('Random Forest Model\'s F1 Score : {:.2f}'.format(rf_f1))
rf_precision = precision_score(test_target, rf_predict)*100
print('Random Forest Model\'s Precision Score : {:.2f}'.format(rf_precision))
rf_recall = recall_score(test_target, rf_predict)*100
print('Random Forest Model\'s Recall Score : {:.2f}'.format(rf_recall))

# Implementing the Support Vector Machine Classifier(SVC) Model
svc_classifier = SVC()
svc_classifier.fit(train_features, train_target)
svc_predict = svc_classifier.predict(test_features)

# It's time to Measure the performance of the Support Vector Machine Classifier(SVC) Model
svc_cm = confusion_matrix(test_target, svc_predict)
print('Support Vector Machine Classifier(SVC) Model\'s Confusion Matrix : {}'.format(svc_cm))
svc_accuracy = accuracy_score(test_target, svc_predict)*100
print('Support Vector Machine Classifier(SVC) Model\'s Accuracy Score : {:.2f}'.format(svc_accuracy))
svc_f1 = f1_score(test_target, svc_predict)*100
print('Support Vector Machine Classifier(SVC) Model\'s F1 Score : {:.2f}'.format(svc_f1))
svc_precision = precision_score(test_target, svc_predict)*100
print('Support Vector Machine Classifier(SVC) Model\'s Precision Score : {:.2f}'.format(svc_precision))
svc_recall = recall_score(test_target, svc_predict)*100
print('Support Vector Machine Classifier(SVC) Model\'s Recall Score : {:.2f}'.format(svc_recall))

# Implementing the Linear Support Vector Classifier(LinearSVC) Model
lsvc_classifier = LinearSVC()
lsvc_classifier.fit(train_features, train_target)
lsvc_predict = lsvc_classifier.predict(test_features)

# It's time to Measure the performance of the Support Vector Machine Classifier(SVC) Model
lsvc_cm = confusion_matrix(test_target, lsvc_predict)
print('LinearSVC Model\'s Confusion Matrix : {}'.format(lsvc_cm))
lsvc_accuracy = accuracy_score(test_target, lsvc_predict)*100
print('LinearSVC Model\'s Accuracy Score : {:.2f}'.format(lsvc_accuracy))
lsvc_f1 = f1_score(test_target, lsvc_predict)*100
print('LinearSVC Model\'s F1 Score : {:.2f}'.format(lsvc_f1))
lsvc_precision = precision_score(test_target, lsvc_predict)*100
print('LinearSVC Model\'s Precision Score : {:.2f}'.format(lsvc_precision))
lsvc_recall = recall_score(test_target, lsvc_predict)*100
print('LinearSVC Model\'s Recall Score : {:.2f}'.format(lsvc_recall))

# Implementing the Perceptron Model
p_classifier = Perceptron()
p_classifier.fit(train_features, train_target)
p_predict = p_classifier.predict(test_features)

# It's time to Measure the performance of the Perceptron Model
p_cm = confusion_matrix(test_target, p_predict)
print('Perceptron Model\'s Confusion Matrix : {}'.format(p_cm))
p_accuracy = accuracy_score(test_target, p_predict)*100
print('Perceptron Model\'s Accuracy Score : {:.2f}'.format(p_accuracy))
p_f1 = f1_score(test_target, p_predict)*100
print('Perceptron Model\'s F1 Score : {:.2f}'.format(p_f1))
p_precision = precision_score(test_target, p_predict)*100
print('Perceptron Model\'s Precision Score : {:.2f}'.format(p_precision))
p_recall = recall_score(test_target, p_predict)*100
print('Perceptron Model\'s Recall Score : {:.2f}'.format(p_recall))

# Implementing the Gaussion Naive Bayes Model
gnb_classifier = GaussianNB()
gnb_classifier.fit(train_features, train_target)
gnb_predict = gnb_classifier.predict(test_features)

# It's time to Measure the performance of the Gaussion Naive Bayes Model
gnb_cm = confusion_matrix(test_target, gnb_predict)
print('Gaussion Naive Bayes Model\'s Confusion Matrix : {}'.format(gnb_cm))
gnb_accuracy = accuracy_score(test_target, gnb_predict)*100
print('Gaussion Naive Bayes Model\'s Accuracy Score : {:.2f}'.format(gnb_accuracy))
gnb_f1 = f1_score(test_target, gnb_predict)*100
print('Gaussion Naive Bayes Model\'s F1 Score : {:.2f}'.format(gnb_f1))
gnb_precision = precision_score(test_target, gnb_predict)*100
print('Gaussion Naive Bayes Model\'s Precision Score : {:.2f}'.format(gnb_precision))
gnb_recall = recall_score(test_target, gnb_predict)*100
print('Gaussion Naive Bayes Model\'s Recall Score : {:.2f}'.format(gnb_recall))

# Implementing the Stochastic Gradient Descent (SGDClassifier) Model
sgd_classifier = SGDClassifier()
sgd_classifier.fit(train_features, train_target)
sgd_predict = sgd_classifier.predict(test_features)

# It's time to Measure the performance of the Stochastic Gradient Descent (SGDClassifier) Model
sgd_cm = confusion_matrix(test_target, sgd_predict)
print('SGDClassifier Model\'s Confusion Matrix : {}'.format(sgd_cm))
sgd_accuracy = accuracy_score(test_target, sgd_predict)*100
print('SGDClassifier Model\'s Accuracy Score : {:.2f}'.format(sgd_accuracy))
sgd_f1 = f1_score(test_target, sgd_predict)*100
print('SGDClassifier Model\'s F1 Score : {:.2f}'.format(sgd_f1))
sgd_precision = precision_score(test_target, sgd_predict)*100
print('SGDClassifier Model\'s Precision Score : {:.2f}'.format(sgd_precision))
sgd_recall = recall_score(test_target, sgd_predict)*100
print('SGDClassifier Model\'s Recall Score : {:.2f}'.format(sgd_recall))

# Implementing the Gradient Boosting Classifier Model
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(train_features, train_target)
gb_predict = gb_classifier.predict(test_features)

# It's time to Measure the performance of the Gradient Boosting Classifier Model
gb_cm = confusion_matrix(test_target, gb_predict)
print('Gradient Boosting Classifier Model\'s Confusion Matrix : {}'.format(gb_cm))
gb_accuracy = accuracy_score(test_target, gb_predict) * 100
print('Gradient Boosting Classifier Model\'s Accuracy Score : {:.2f}'.format(gb_accuracy))
gb_f1 = f1_score(test_target, gb_predict) * 100
print('Gradient Boosting Classifier Model\'s F1 Score : {:.2f}'.format(gb_f1))
gb_precision = precision_score(test_target, gb_predict) * 100
print('Gradient Boosting Classifier Model\'s Precision Score : {:.2f}'.format(gb_precision))
gb_recall = recall_score(test_target, gb_predict) * 100
print('Gradient Boosting Classifier Model\'s Recall Score : {:.2f}'.format(gb_recall))

# Model's Performance based on Accuracy Score
models_performance = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 
              'Linear SVC',  'Perceptron', 'Naive Bayes', 'Stochastic Gradient Decent', 'Gradient Boosting'],
    'Accuracy Score': [lgr_accuracy, knn_accuracy, dt_accuracy, rf_accuracy, svc_accuracy, 
              lsvc_accuracy, p_accuracy, gnb_accuracy, sgd_accuracy, gb_accuracy],
    'f1 Score': [lgr_f1, knn_f1, dt_f1, rf_f1, svc_f1, lsvc_f1, p_f1, gnb_f1, sgd_f1, gb_f1],
    'Precision Score': [lgr_precision, knn_precision, dt_precision, rf_precision, svc_precision,
                        lsvc_precision, p_precision, gnb_precision, sgd_precision, gb_precision],
    'Recall Score': [lgr_recall, knn_recall, dt_recall, rf_recall, svc_recall, lsvc_recall,
                     p_recall, gnb_recall, sgd_recall, gb_recall]})
# models = models.sort_values(by=['Accuracy Score', 'f1 Score', 'Precision Score', 'Recall Score'], ascending=False)
models_performance = models_performance.sort_values(by='Accuracy Score', ascending=False)

# Cross Validation Score for the SVC Model
svc_cvs = cross_val_score(estimator = svc_classifier, X = train_features, y = train_target, cv = 10, scoring = 'accuracy')
print('Accuracy\'s Mean of SVC Model : {:.2f}'.format(svc_cvs.mean()*100))
print('Accuracy\'s Standard Deviation of SVC Model : {:.2f}'.format(svc_cvs.std()*100))

# Its time to check the kernel performance and its classification report
svc_class_report = classification_report(test_target, svc_predict)
print(svc_class_report)

kernels = ['linear' ,'polynomial', 'rbf', 'sigmoid']
def get_classifier(ktype):
    if ktype == 0:
        return SVC(kernel = 'linear', gamma = 'auto')
    if ktype == 1:
        return SVC(kernel = 'poly', degree = 10, gamma = 'auto')
    if ktype == 2:
        return SVC(kernel = 'rbf', gamma = 'auto')
    if ktype == 3:
        return SVC(kernel = 'sigmoid', gamma = 'auto')

for i in range(4):
    svc_classifier_ktype = get_classifier(i)
    svc_classifier_ktype.fit(train_features, train_target)
    svc_classifier_ktype_predict = svc_classifier_ktype.predict(test_features)
    print('Evaluation: ', kernels[i], 'Kernel')
    print(classification_report(test_target, svc_classifier_ktype_predict))
    
# It's time to tune the hyper-parameters for SVC model (Tunning for Kernel, C(Regularisation) and gamma)
parameters = {'kernel': ['rbf', 'sigmoid', 'poly'],
              'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), parameters, refit = True, verbose = 2)
grid.fit(train_features, train_target)

print(grid.best_estimator_)