"""
This script contains the commands used to perform statistical analyses on the database.
Specifically, it includes tests for correlations between variables and for assessing
relationships between numerical and categorical features.
"""

############################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# February 2026

#############################################################################################################

from functions import *              # it contains utility functions
import pandas as pd                  # it permits data manipulation and analysis
import numpy as np                   # it is a package for scientific computing in Python
from sklearn import preprocessing    # used for standardization
import seaborn as sns                # it contains tools for statistical data visualization
import matplotlib.pyplot as plt      # it is a library for creating plots

if __name__ == '__main__':  

    dirpath = "..."
    file = "..."
    sheet_name = "..."

    # Divide numerical and categorical features
    relevant_columns = [...]

    numerical = [...]
    categorical = [...]

    # Read the Excel file and convert categorical features to numeric
    dataset = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name, converters={feature: functions_for_converting})
    
    dataset = dataset[relevant_columns].copy()
   

    # Correlation plot. Spearman correlation was used to compute the correlations in the feature matrix, and features with strong correlations were arranged close together in the plot
    dataset_numerical = dataset[numerical].copy()

    corr = dataset_numerical.corr(method = 'spearman')
    corr = corr.fillna(0)

    sns.set(font_scale=0.6)
    fig1 = sns.clustermap(corr, method='ward', xticklabels=True, yticklabels=True)
    fig1.ax_row_dendrogram.set_visible(False)
    fig1.ax_col_dendrogram.set_visible(False)
    plt.title('correlation plot')
    fig1.savefig(dirpath +"correlation_plot.png")
    plt.show()

    # Compute correlation with p-values

    with open(dirpath +"Correlations.txt", "w") as f:
        f.write("                   Spearman's correlation  | Kendall's correlation  \n")
        f.write("----------------------------------------------------------------------------------------\n")

        for ii in range(len(numerical)):
            feat1 = numerical[ii]
            for jj in range(ii+1, len(numerical)):
                feat2 = numerical[jj]
                S, Spvalue, K, Kpvalue = correlation_pvalue(dataset_numerical,  feat1, feat2)
                f.write(f"                %s - %s      &  %1.3f (p-value %1.2e)  &  %1.3f (p-value %1.2e)  \\\\\n"  %(feat1, feat2, S, Spvalue, K, Kpvalue))


    # Test with categorical features
    with open(dirpath + "Relations.txt", "w") as f:
        f.write("  feat categorical -  feat      |      method      |  stat   p-value     \n")
        for feat_cat in categorical:
            for feat in dataset.columns:
                if feat != feat_cat:
                    if feat in numerical:
                        method = 'Kruskal-Wallis'
                        stat, p = Kruskal_Wallis(dataset, feat_cat, feat)
                    else:
                        method = 'Chi-quadro'
                        stat, p = Chi_quadro(dataset, feat_cat, feat)
                    f.write(f"   %s - %s & %s &  %1.3f (p-value %1.2e)  \\\\\n" %(feat_cat, feat, method, stat, p))

