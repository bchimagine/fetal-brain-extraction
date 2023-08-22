import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

data = pd.DataFrame()
path = "../MonaiBased/results/*.csv"
for fname in glob.glob(path):
    data_ = pd.read_csv(fname, on_bad_lines='skip')
    data = pd.concat([data, data_])

############################################################

data_t2 = data[data['Modality'] == 'fMRI']
# data = data.drop(columns=['Type'])
data_subjwise = (data_t2.groupby(['Method', 'Modality', 'Type', 'Subject']).
                 agg({'Dice': 'mean', 'Sensitivity': 'mean', 'Specificity': 'mean'}).reset_index())
sns.set_theme()
sns.set_style("whitegrid")
g = sns.boxplot(data=data_subjwise, x='Modality', y='Dice', hue='Type', showmeans=True,
                flierprops=dict(markerfacecolor='0.50', markersize=1)
                )
g.set(ylim=(0, 1))
plt.show()

############################################################
data = data[data['Modality'] != 'otherscanners']
# data = data.drop(columns=['Type'])
data_subjwise = (data.groupby(['Method', 'Modality', 'Type', 'Subject']).
                 agg({'Dice': 'mean', 'Sensitivity': 'mean', 'Specificity': 'mean'}).reset_index())


# # fig, axes = plt.subplots(1, 1, figsize=(6, 6))
# # fig.suptitle('Pokemon Stats by Generation')
sns.set_theme()
sns.set_style("whitegrid")
g = sns.boxplot(data=data_subjwise, x='Modality', y='Dice', hue='Method', showmeans=True,
                flierprops=dict(markerfacecolor='0.50', markersize=1)
                # meanprops={"marker":"s",
                #            "markerfacecolor":"white",
                #            "markeredgecolor":"black",
                #           "markersize":"6"}
                )
g.set(ylim=(0, 1))
# # plt.yscale('log')
# # sns.boxplot(ax=axes[0, 1], data=new_t2, x='modality', y='dice', hue='type', showmeans=True)
# # sns.boxplot(ax=axes[0, 2], data=new_data_otherscanners, x='modality', y='dice', hue='type', showmeans=True)
# # sns.boxplot(ax=axes[1, 0], data=new_dwi, x='modality', y='dice', hue='type', showmeans=True)
# # sns.boxplot(ax=axes[1], data=data, x='modality', y='dice', hue='type', showmeans=True)
# # sns.boxplot(ax=axes[1, 2], data=new_t2, x='modality', y='dice', hue='type', showmeans=True)
# # sns.despine()
# # plt.savefig('otherscanners.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
