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
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split



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
    scores = cross_val_score(clf, xtrain, ytrain, cv=6)
    print(scores)
    print(scores.mean())
    exit()
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
    mean_fare = df['fare'].mean()
    df['fare'] = df['fare'].fillna(mean_fare)
    mean_age = df['age'].mean()
    df['age'] = df['age'].fillna(mean_age)
    embarked_port = 'S'
    df['embarked'] = df['embarked'].fillna(embarked_port)
    df.sex = [gender[item] for item in df.sex]
    df.embarked = [embark[item] for item in df.embarked]
    feature_cols = ['pclass', 'sex', 'age',  'survived',  'embarked']
    # x_features = ['pclass', 'sex', 'age', 'sibsp','fare','embarked']
    df = df[feature_cols]
    df2 = df
    classifier = KNeighborsClassifier(n_neighbors=9)
    ytrain = df2.survived
    xtrain = df2.drop(['survived'], axis=1)
    X,Y = xtrain.values,ytrain.values
    scores = cross_val_score(classifier, xtrain, ytrain, cv=6)
    print(scores.mean())
    exit()
    X = X[:,:3]
    # classifier.fit(xtrain, ytrain)
    # result = classifier.predict(xtrain)
    # print("Accuracy:", metrics.accuracy_score(result, list(ytrain)))
    classifier.fit(X, Y)
    result = classifier.predict(X)
    print("Accuracy:", metrics.accuracy_score(result, list(ytrain)))

    n_neighbors = 5
    h = 0.02
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    # calculate min, max and limits
    a_min, a_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    b_min, b_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(b_min,b_max)
    c_min, c_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    print(c_min,c_max)
    # d_min, d_max = X[:, 3].min() - 1, X[:, 3].max() + 1
    # e_min, e_max = X[:, 4].min() - 1, X[:, 4].max() + 1
    # f_min, f_max = X[:, 5].min() - 1, X[:, 5].max() + 1
    xx, yy = np.meshgrid(np.arange(a_min, a_max, h),
                         np.arange(b_min, b_max, h),
                         np.arange(c_min, c_max, h),
                         # np.arange(d_min, d_max, h),
                         # np.arange(e_min, e_max, h),
                         # np.arange(f_min, f_max, h),
                         )

    # predict class using data and kNN classifier
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (n_neighbors))
    plt.show()


def q4():


if __name__ == "__main__":
    df = data_preprocess()
    q2(df)
    # q2_logt_reg()
    # q3(df)
