import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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


def q2_logt_reg():
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
    model_stat = sm.Logit(Y, X).fit()
    ypred = model_stat.predict(X)
    binary_pred = []
    for i in ypred:
        if i >= 0.5:
            binary_pred.append(1)
        else:
            binary_pred.append(0)
    print("Accuracy:", metrics.accuracy_score(binary_pred, list(Y)))


def q2(df):
    x_features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'embarked']
    ytrain = df.survived
    xtrain = df.drop(['survived'], axis=1)
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(xtrain, ytrain)
    print("finished")
    reuslt = clf.predict(xtrain)
    print("Accuracy:", metrics.accuracy_score(reuslt, list(ytrain)))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=x_features, class_names=['Not_survived', 'Survived'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png())

    plt.show()


def q3(df):
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
    classifier = KNeighborsClassifier(n_neighbors=5)
    ytrain = df2.survived
    xtrain = df2.drop(['survived'], axis=1)
    classifier.fit(xtrain, ytrain)
    result = classifier.predict(xtrain)
    print("Accuracy:", metrics.accuracy_score(result, list(ytrain)))


if __name__ == "__main__":
    df = data_preprocess()
    q2(df)
    # q2_logt_reg()
    # q3()
