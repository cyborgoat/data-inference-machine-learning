import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def q3():
    df = pd.read_excel("Diabetes_Data.xlsx")
    pd.set_option('display.max_columns', 30)
    print(df)
    df_corr = df[df.columns[:-1]].corr()
    print(df_corr)
    heatmap = sns.heatmap(df_corr)
    plt.show()








if __name__ == "__main__":
    q3()