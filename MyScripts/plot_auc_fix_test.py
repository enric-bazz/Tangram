
import os

import matplotlib.pyplot as plt
import pandas as pd

import tangram as tg

dir = os.getcwd()
print(dir)
path = '/data/test_df.csv'

# Create genes dataframe
df_all_genes = pd.read_csv(path)
df_crop = df_all_genes.iloc[:len(df_all_genes)//100]

# plot AUC
auc_fig = tg.plot_auc(df_all_genes)
print('figure saved correctly as {}'.format(type(auc_fig)))
print(auc_fig)
plt.show()
plt.close()