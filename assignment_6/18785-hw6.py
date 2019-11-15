import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
import statsmodels.api as sm
from IPython.display import Image
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import export_graphviz
from sklearn.linear_model import LassoCV
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def data_preprocess():
    gender = {'male': 1, 'female': 2}
    embark = {'S': 1, 'C': 2, 'Q': 3}
    df = pd.read_csv("titanic3.csv")
    feature_cols = ['pclass', 'sex', 'age', 'fare', 'survived', 'sibsp', 'embarked']
    x_features = ['pclass', 'sex', 'age', 'sibsp']
    df = df[feature_cols]
    mean_age = df['age'].mean()
    df['age'] = df['age'].fillna(mean_age)
    mean_fare = df['fare'].mean()
    df['fare'] = df['fare'].fillna(mean_fare)
    embarked_port = 'S'
    df['embarked'] = df['embarked'].fillna(embarked_port)
    df.sex = [gender[item] for item in df.sex]
    df.embarked = [embark[item] for item in df.embarked]
    df2 = pd.concat([df], axis=1)
    return df2


def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)


def p2_logt_reg():
    df = pd.read_csv("titanic3.csv")
    sex = pd.get_dummies(df['sex'], drop_first=True)
    pclass = pd.get_dummies(df['pclass'], drop_first=True)
    # embark = pd.get_dummies(df['embarked'], drop_first = True)
    df.drop(['sex', 'embarked', 'name', 'ticket', 'body', 'home.dest'], axis=1, inplace=True)
    train = pd.concat([df, sex, pclass], axis=1)
    pd.set_option('display.max_columns', None)
    X = train.drop(['pclass', 'sibsp', 'parch', 'fare', 'cabin', 'boat'], axis=1)
    X = X.dropna()
    print(X.columns)
    Y = np.array(X['survived'])
    X.drop(['survived'], axis=1, inplace=True)
    model_stat = LogisticRegression().fit(X,Y)
    ypred = model_stat.predict(X)
    binary_pred = []
    for i in ypred:
        if i >= 0.5:
            binary_pred.append(1)
        else:
            binary_pred.append(0)
    print("Accuracy:", metrics.accuracy_score(binary_pred, list(Y)))
    print("Cross Validation Score: ",cross_val_score(LogisticRegression(), X, Y, cv=10).mean())


class p2():
    def __init__(self, df):
        self.x_features = ['pclass', 'sex', 'age']
        self.ytrain = df.survived
        self.xtrain = df[self.x_features]

    def q3(self):
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(self.xtrain, self.ytrain)
        print("finished")
        reuslt = clf.predict(self.xtrain)
        scores = cross_val_score(clf, self.xtrain, self.ytrain, cv=10)
        print(scores)
        print(scores.mean())
        print("Accuracy:", metrics.accuracy_score(reuslt, list(self.ytrain)))

        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=self.x_features, class_names=['Not_survived', 'Survived'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('tree.png')
        Image(graph.create_png())
        plt.show()

    def q4(self):
        clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                                     max_features=None, max_leaf_nodes=15, min_samples_leaf=3,
                                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=None, splitter='random')
        clf = clf.fit(self.xtrain, self.ytrain)
        print("finished")
        reuslt = clf.predict(self.xtrain)
        scores = cross_val_score(clf, self.xtrain, self.ytrain, cv=6)
        print(scores)
        print(scores.mean())
        print("Accuracy:", metrics.accuracy_score(reuslt, list(self.ytrain)))


def p3(df):
    gender = {'male': 1, 'female': 2}
    embark = {'S': 1, 'C': 2, 'Q': 3}
    df = pd.read_csv("titanic3.csv")
    feature_cols = ['pclass', 'sex', 'age', 'survived']
    mean_fare = df['fare'].mean()
    df['fare'] = df['fare'].fillna(mean_fare)
    mean_age = df['age'].mean()
    df['age'] = df['age'].fillna(mean_age)
    embarked_port = 'S'
    df['embarked'] = df['embarked'].fillna(embarked_port)
    df.sex = [gender[item] for item in df.sex]
    df.embarked = [embark[item] for item in df.embarked]
    feature_cols = ['pclass', 'sex', 'age', 'survived']
    # x_features = ['pclass', 'sex', 'age', 'sibsp','fare','embarked']
    df = df[feature_cols]
    df2 = df
    result = []
    acc = []
    for i in range(2,11):
        classifier = KNeighborsClassifier(n_neighbors=i)
        ytrain = df2.survived
        xtrain = df2.drop(['survived'], axis=1)
        X, Y = xtrain.values, ytrain.values
        scores = cross_val_score(classifier, X, Y, cv=10)
        result.append(scores.mean())
        model = classifier.fit(X,Y)
        y_pred = model.predict(X)
        acc.append(metrics.accuracy_score(y_true=Y,y_pred=y_pred))
    print("Accuracy:",acc)
    print("CrossVal:",result)
    plt.plot(np.arange(2,11),result,label='10-Fold Cross Validation Score')
    plt.plot(np.arange(2,11),acc,label='In sample Accuracy')
    plt.xlabel("# of Neigbhors")
    plt.ylabel("Performance")
    plt.legend()
    plt.show()
    # exit()
    # Different Metrics
    classifier = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("eucledian", scores.mean())
    classifier = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("manhattan", scores.mean())

    classifier = KNeighborsClassifier(n_neighbors=3, metric="chebyshev")
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("chebyshev", scores.mean())

    classifier = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("minkowski", scores.mean())

    # classifier = KNeighborsClassifier(n_neighbors=6,metric="wminkowski")
    # scores = cross_val_score(classifier, xtrain, ytrain, cv=6)
    # print("wminkowski",scores.mean())

    # classifier = KNeighborsClassifier(n_neighbors=6,metric="seuclidean")
    # scores = cross_val_score(classifier, xtrain, ytrain, cv=6)
    # print("seuclidean",scores.mean())

    classifier = KNeighborsClassifier(n_neighbors=6, metric="mahalanobis", metric_params={'V': np.cov(X)})
    scores = cross_val_score(classifier, X, Y, cv=10)
    print("mahalanobis", scores.mean())




class p4():
    def __init__(self):
        self.df_red = pd.read_csv("winequality-red.csv", sep=';')
        self.df_white = pd.read_csv("winequality-white.csv", sep=';')

    def q1(self):
        print("---red---")
        x_features = list(self.df_white.columns)
        red_mean = list(self.df_red.mean())
        white_mean = list(self.df_white.mean())
        # ax = plt.subplot(111)
        x = np.arange(len(x_features))

        # print(x)
        # print(red_mean)
        # print(white_mean)
        plt.bar(x + 0.2, red_mean, width=0.2, color='red', align='center', label='Red wine')
        plt.bar(x, white_mean, width=0.2, color='grey', align='center', label='White wine')
        print(x_features)
        plt.xticks(x, x_features, rotation=90)
        plt.legend()
        plt.show()

    def q2(self):
        cor_white = self.df_white.corr()
        cor_red = self.df_red.corr()
        # print(cor_red["quality"])
        print(cor_white["quality"])
        sns.heatmap(cor_white, annot=True, cmap=plt.cm.Reds)
        plt.title("Correlation heatmap of white wine")
        plt.show()
        sns.heatmap(cor_red, annot=True, cmap=plt.cm.Reds)
        plt.title("Correlation heatmap of red wine")
        plt.show()

    def q3(self):
        Xw, Yw = self.df_white.drop(['quality'], axis=1), self.df_white['quality']
        Xr, Yr = self.df_red.drop(['quality'], axis=1), self.df_red['quality']

        def get_wine_info(x, y, kind):
            clf = LassoCV(cv=5)
            model = clf.fit(x, y)
            mse = [i.mean() for i in model.mse_path_]
            lambdas = model.alphas_
            plt.plot(lambdas, mse)
            plt.xlabel("lambda")
            plt.ylabel("MSE")
            plt.title("{} Wine MSE vs. Lambda".format(kind))
            plt.show()
            params = model.coef_
            print(params)
            sfm = SelectFromModel(clf, threshold=0.25)
            sfm.fit(x, y)
            n_features = sfm.transform(x).shape[1]
            X_transform = sfm.transform(x)
            print("----{} Wine-----".format(kind))
            for i, j in zip(list(x.columns), sfm.get_support()):
                print("{} : {}".format(i, j))

        get_wine_info(Xw, Yw, "white")
        get_wine_info(Xr, Yr, "Red")

    def q4(self):
        x, y = self.df_red[['volatile acidity', 'sulphates', 'alcohol']], self.df_red[['quality']]
        y = y.values.ravel()
        classifier = KNeighborsRegressor(n_neighbors=2, metric='euclidean')
        model = classifier.fit(x, y)
        print(model.get_params)
        # print(model.kneighbors_graph(x))
        scores = cross_val_score(classifier, x, y, cv=6)
        y_pred = model.predict(x)
        binary_pred = []
        acc = metrics.accuracy_score(y_true=y, y_pred=[int(i) for i in y_pred])
        print("Cross Validation Score:", scores.mean())
        print("Accuracy: ", acc)
        # classifier =

    def q5(self):
        x, y = self.df_red[['volatile acidity', 'sulphates', 'alcohol']], self.df_red[['quality']]
        y = y.values.ravel()
        classifier = KNeighborsRegressor(n_neighbors=5)
        model = classifier.fit(x, y)
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        R_sqrd = r2_score(y, y_pred=y_pred)
        print("MSE for KNN Regression: ", mse)
        print("R^2 for KNN Regression: ", R_sqrd)
        classifier = LinearRegression()
        model = classifier.fit(x, y)
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        R_sqrd = r2_score(y, y_pred=y_pred)
        print("MSE for Linear Regression: ", mse)
        print("R^2 for Linear Regression: ", R_sqrd)


if __name__ == "__main__":
    df = data_preprocess()
    # p2(df).q3()
    # p2_logt_reg()
    p3(df)
    # p4().q2()
    # p4().q3()
    # p4().q4()
    # p4().q5()
