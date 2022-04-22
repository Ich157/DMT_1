import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway


sns.set_theme(style="ticks", palette="pastel")
# correlation heatmap
train_df = pd.read_csv("train.csv")
train_df = train_df.drop(columns=['Unnamed: 0.1','Unnamed: 0'], axis=1)

#ax = sns.heatmap(train_df.corr())
sns.despine()
plt.tight_layout()
#plt.savefig("corr_heatmap.pdf")
#sns.scatterplot(train_df['social'], train_df['communication'])
#plt.show()

#print(train_df['mood'].corr(train_df['valence']))



# boxplot
# TODO: make dataframe out of the results for the models with column name is the model and the values are the MAE values.

#lstm = read_csv("")
# get knn error
k = open("knn_errors.txt", "r").read()
knn = k.split("\n")
knn.pop()
knn = [float(x) for x in knn]
knn = np.array(knn)
knn_log = np.log(knn)
# get tree error
t = open("tree_errors.txt", "r").read()
tree = t.split("\n")
tree.pop()
tree = [float(x) for x in tree]
tree = np.array(tree)
tree_log = np.log(tree)
# get lstm error
lstm = pd.read_csv("error_LSTM.csv")
lstm = lstm.drop(['Unnamed: 0'], axis=1)



results = {'KNN Algorithm': knn, 'Tree Regression': tree}
df = pd.DataFrame(data=results)
df['LSTM'] = lstm
print(df)
df_melt = pd.melt(df)
p = sns.boxplot(x="variable", y="value", data=pd.melt(df))
p.set_xlabel("Model", fontsize=10)
p.set_ylabel("Mean Absolute Error (MAE)", fontsize=10)
p.set(ylim=(-0.1, 2.25))
sns.despine()
plt.tight_layout()
plt.savefig("results.pdf")
plt.show()

# table
#TODO: make table with mean, sd for each model
knn_mean = np.mean(np.array(knn))
knn_sd = np.std(np.array(knn))
knn_med = np.median(np.array(knn))
tree_mean = np.mean(np.array(tree))
tree_sd = np.std(np.array(tree))
tree_med = np.median(np.array(tree))
lstm_mean = np.mean(np.array(lstm))
lstm_sd = np.std(np.array(lstm))
lstm_med = np.median(np.array(lstm))


#TODO: statistical test (anova) with https://www.reneshbedre.com/blog/anova.html
f_val, p_val = f_oneway(df['KNN Algorithm'], df['Tree Regression'], df['LSTM'])
#print(f_val, p_val)  # 2.211, 0.11 -> not significant

model = ols('value ~ C(variable)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)