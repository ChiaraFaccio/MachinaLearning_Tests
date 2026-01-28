"""
The script computes exploratory data analysis for times series
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

    dirpath_img = ""
    file = ""
    dirpath = ""

    # Read the Excel file 
    dataset = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name, converters={feature: functions_for_converting})    

    # Features used to compute clusters
    relevant_features = ['time_point'] + [...]
    
    dataset = dataset[relevant_features].copy()
    for ii in relevant_features[1:]:
        dataset[ii] = pd.to_numeric(dataset[ii], errors = "coerce")
    
    x = ['T0', 'T3', 'T6', 'T9', 'T12']    # we suppose having 5 time points : T0, T3, T6, T9, T12
    
    analysis = [...] # list of features we want to analyze

    for feat in analysis:
        fig = plt.figure()
            
        median_phen = []
        for ii in x:
            dataset_time = dataset[dataset['time_point'] == ii].copy()
            median_phen.append(dataset_time[feat].median())        

        plt.subplot(1, 1, 1)
        g = sns.boxplot(data=dataset, x = 'time_point', y=feat)
        plt.plot(x, median_phen, color='red')
        plt.title(feat)
        plt.ylim([min(np.array(dataset[feat])),max(np.array(dataset[feat]))+2])
        plt.savefig(dirpath_img + feat + '_across_time.png')
        plt.show()