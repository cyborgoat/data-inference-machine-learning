import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            # print("-------------------")
            # print(model.summary())
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add Variable:  {:10} | p-value {:.5}'.format(best_feature, best_pval))

        # backward step
        # model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # # print("-------------------")
        # # print(model.summary())
        # # use all coefs except intercept
        # pvalues = model.pvalues.iloc[1:]
        # worst_pval = pvalues.max() # null if pvalues is empty
        # if worst_pval > threshold_out:
        #     changed=True
        #     worst_feature = pvalues.argmax()
        #     included.remove(worst_feature)
        #     if verbose:
        #         print('Drop {:30} with p-value {:.5}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included,model

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def q3():
    # 3.1
    df = pd.read_excel("Diabetes_Data.xlsx")
    pd.set_option('display.max_columns', 30)
    # print(df)
    df_corr = df[df.columns[:-1]].corr()
    print(df_corr)
    heatmap = sns.heatmap(df_corr)
    plt.show()
    # 3.3
    X = df[df.columns[:-1]]
    Y = df['Y']
    # model = linear_model.LinearRegression()
    # model.fit(X,Y)
    model = sm.OLS(Y, X)
    model = model.fit()
    print(model.summary())
    # print('Intercept: \n', model.intercept_)
    # print('Coefficients: \n', model.coef_)
    result = model.predict(X)
    mse = mean_squared_error(Y, result)
    print(mse)
    #3.5
    sw_features,sw_model = stepwise_selection(X=X,y=Y)
    # mse = mean_squared_error(Y, sw_model.predict(X))
    print(sw_features)
    print(sw_model.summary())
    X['const']=1
    sw_ypred = sw_model.predict(X[['const','BMI', 'S5', 'BP', 'S1', 'SEX', 'S2','AGE']])
    sw_mse = mean_squared_error(Y, sw_ypred)
    print(sw_mse)


def q4():
    df = pd.read_csv("titanic3.csv")

    def get_surviveprob(cur_df):
        survived = (cur_df['survived'] == 1).sum()
        not_survived = (cur_df['survived'] == 0).sum()
        return (survived / (survived + not_survived))

    # print(df)
    # 4.2
    tot_survive_prob = get_surviveprob(df)
    print("Titanic survival probability is :{}".format(tot_survive_prob))
    # 4.3
    # Children (00-14 years), Youth (15-24 years), Adults (25-64 years), Seniors (65 years and over)
    df3 = df[['age', 'survived']]
    df3 = df3.loc[(df3.age.notnull()) & (df3.survived.notnull())]
    child_stat = df3[df3['age'].between(0, 14, inclusive=True)]
    youth_stat = df3[df3['age'].between(15, 24, inclusive=True)]
    adults_stat = df3[df3['age'].between(25, 64, inclusive=True)]
    senior_stat = df3[df3['age'] > 64]
    child_sp = get_surviveprob(child_stat)
    youth_sp = get_surviveprob(youth_stat)
    adults_sp = get_surviveprob(adults_stat)
    senior_sp = get_surviveprob(senior_stat)
    print("Children (00-14 years), Youth (15-24 years), Adults (25-64 years), Seniors (65 years and over)")
    print("-----Survival Probability by Age Class-----\nChildren:{}\nYouth:{}\nAdults:{}\nSenior:{}".format(child_sp, youth_sp, adults_sp, senior_sp))
    pclass = df['pclass'].unique()
    # print(pclass)
    pclass_stat, pclass_sp = [], []
    for i in pclass:
        pclass_stat.append(df[df['pclass'] == i])
    print("\n--------Survival Probability by Passenger Class-----")
    for i, j in enumerate(pclass_stat):
        print("class:{} | survive prob:{}".format(i + 1, get_surviveprob(j)))
    gen_class = df['sex'].unique()
    # print(pclass)
    genclass_stat, genclass_sp = [], []
    for i in gen_class:
        genclass_stat.append(df[df['sex'] == i])
    print("\n--------Survival Probability by Gender-----")
    for i, j in enumerate(genclass_stat):
        print("class:{} | survive prob:{}".format(gen_class[i], get_surviveprob(j)))
    print("\n--------------------\n--------Q4P4--------\n")
    # 4.4
    sex = pd.get_dummies(df['sex'], drop_first=True)
    pclass = pd.get_dummies(df['pclass'], drop_first=True)
    # embark = pd.get_dummies(df['embarked'], drop_first = True)
    df.drop(['sex', 'embarked', 'name', 'ticket', 'body', 'home.dest'], axis=1, inplace=True)
    train = pd.concat([df, sex, pclass], axis=1)
    pd.set_option('display.max_columns', None)
    X = train.drop(['pclass', 'sibsp', 'parch', 'fare', 'cabin', 'boat'], axis=1)
    X = X.dropna()
    Y = np.array(X['survived'])
    X.drop(['survived'], axis=1, inplace=True)
    model_stat = sm.Logit(Y, X).fit()
    ypred= model_stat.predict(X)
    # print(len(ypred),len(X),len(Y))
    # print(Y)
    binary_pred = []
    for i in ypred:
        if i>=0.5:
            binary_pred.append(1)
        else:
            binary_pred.append(0)

    print(model_stat.summary())
    # 4.5
    cm = confusion_matrix(y_true=Y, y_pred=binary_pred, labels=[0, 1])
    correct = 0
    for i in range(len(Y)):
        if Y[i] == binary_pred[i]:
            correct+=1
    print(correct/len(Y))
    # heatmap = sns.heatmap(cm,annot=True)
    # plt.imshow(cm, cmap='hot', interpolation='nearest')
    # plt.show()
    ax = sns.heatmap(cm, linewidth=0.5)
    plt.xlabel("Predicted Survival Statistics \n(0: Not survived 1: Survived)")
    plt.ylabel("Actual Survival Statistics \n(0: Not survived 1: Survived)")
    plt.show()


if __name__ == "__main__":
    # q3()
    q4()
