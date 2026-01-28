"""
The script computes exploratory data analysis 
"""

############################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
# February 2026

#############################################################################################################

from functions import *                            # it contains utility functions
import pandas as pd                                # it permits data manipulation and analysis
import numpy as np                                 # it is a package for scientific computing in Python
import seaborn as sns                              # it contains tools for statistical data visualization
import matplotlib.pyplot as plt                    # it is a library for creating plots

################################################################################################################



if __name__ == '__main__':  

    dirpath = "..."
    dirpath_save = "..."   # to save
    file = "..."
    sheet_name = "..."

    # Read the Excel file and convert categorical features to numeric
    X = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name, converters={feature: functions_for_converting})
    columns = X.columns
    
    # Divide numerical and categorical features
    numerical = [...]
    categorical = [...]

    X_numerical = X[numerical].copy()
    X_categorical = X[categorical].copy()

    print(f'There are {len(numerical)} numerical features: \n')
    print(numerical)
    print(f'There are {len(categorical)} categorical features: \n')
    print(categorical)
    
    # Count the percentage of missing values
    print(f'*** Percentage of missing values: ***')
    for ii in columns:
        print("{} -> {:3.2f}%".format(ii, (1 - X[ii].count()/len(X))*100))
    
    # Basic statistic information for numerical features
    print(f'\n*** Basic statistic infomìrmation for numerical featires: ***')
    print(X_numerical.describe().T)
    
    # Basic statistic information for categorical features
    print(f'\n*** Basic statistic infomìrmation for categorical featires: ***')
    print(X_categorical.describe(include = 'object').T)
    
    
    # Univariate graphical analysis for numerical features (box plot, violin plot and histogram)
    plot_numerical_features(X_numerical, dirpath_save)
    
    # Univariate graphical analysis for categorical features (count plot)
    plot_categorical_features(X_categorical, dirpath_save)
    
    
    # Bivariate analysis : correlations
    corr = X_numerical.corr(method = 'spearman')
    corr = corr.fillna(0)

    # Multivariate analysis: Spearman correlation was used to compute the correlations in the feature matrix, and features with strong correlations were arranged close together in the plot
    sns.set(font_scale=0.6)
    fig1 = sns.clustermap(corr, method='ward', xticklabels=True, yticklabels=True)
    fig1.ax_row_dendrogram.set_visible(False)
    fig1.ax_col_dendrogram.set_visible(False)
    plt.title('correlation plot')
    fig1.savefig(dirpath_save +"correlation_plot.png")
    plt.show()

    # Compute correlation with p-values
    with open(dirpath_save +"Correlations.txt", "w") as f:
        f.write("                                      Spearman's correlation  | Kendall's correlation  \n")
        f.write("-----------------------------------------------------------------------------------------------------------\n")

        for ii in range(len(numerical)):
            feat1 = numerical[ii]
            for jj in range(ii+1, len(numerical)):
                feat2 = numerical[jj]
                S, Spvalue, K, Kpvalue = correlation_pvalue(X_numerical,  feat1, feat2)
                f.write(f"                %s - %s      &  %1.3f (p-value %1.2e)  &  %1.3f (p-value %1.2e)  \n"  %(feat1, feat2, S, Spvalue, K, Kpvalue))


    # Test with categorical features
    with open(dirpath_save + "Relations.txt", "w") as f:
        f.write("  feat categorical -  feat      |      method      |  stat   p-value     \n")
        for feat_cat in categorical:
            for feat in columns:
                if feat != feat_cat:
                    if feat in numerical:
                        method = 'Kruskal-Wallis'
                        stat, p = Kruskal_Wallis(X, feat_cat, feat)
                    else:
                        method = 'Chi-quadro'
                        stat, p = Chi_quadro(X, feat_cat, feat)
                    f.write(f"   %s - %s & %s &  %1.3f (p-value %1.2e)  \n" %(feat_cat, feat, method, stat, p))
        
    
    
    