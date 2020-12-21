import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.test_data = pd.read_csv(test_data_path)

    def predict(self):
        X_train,y_train = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        level_0 = list()
        level_0.append(('RF', RandomForestClassifier(n_estimators=700)))
        level_0.append(('LR',LogisticRegression(max_iter=6000)))
        
        level_1 = SVC(C=1.2)
        model = StackingClassifier(estimators=level_0, final_estimator=level_1, cv=4)

        model.fit(X_train, y_train)
        test=scaler.transform(self.test_data)
        submission = model.predict(test)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)
