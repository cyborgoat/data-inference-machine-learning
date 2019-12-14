import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from data_preprocess import train,test,features_drop

if __name__ == "__main__":
    train = train.drop(features_drop, axis=1)
    test = test.drop(features_drop, axis=1)
    train = train.drop(['PassengerId'], axis=1)
    train_data = train.drop('Survived', axis=1)
    target = train['Survived']
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = SVC()
    clf.fit(train_data, target)
    scoring = 'accuracy'
    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    print(score)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('titanic_submission.csv', index=False)
