# This script contains various functions useful for hte experiments

###############################################################################################################################

#!/usr/bin/env python3

# Chiara Faccio, Università degli Studi di Padova
# chiara.faccio@unipd.it
#Febraury 2026

################################################################################################################################

from scipy.stats import spearmanr, kendalltau

from sklearn import preprocessing
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Leiden
from sknetwork.clustering import get_modularity
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.linalg import issymmetric

from scipy.stats import kruskal, chi2_contingency

from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import mutual_info_classif
from feature_engine.selection import MRMR
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE

############################################################################################################################################


# Function to compute Spearman and Kendall correlations and p-values

def correlation_pvalue(data, labelx, labely):

	# spearmanr and kendalltau do not support missing value
	tmp = data[data[labelx].notna() & data[labely].notna()]
	
	# Spearman's     
	S, Spvalue = spearmanr(tmp[labelx],tmp[labely])	
	
	# Kendall's 
	K, Kpvalue = kendalltau(tmp[labelx],tmp[labely])

	return S, Spvalue, K, Kpvalue



# Functions to compute Kruskal–Wallis

def Kruskal_Wallis(data, feat_categorical, feat_numerical):

	groups = [group[feat_numerical].dropna() for _, group in data.groupby(feat_categorical)]
	stat, p = kruskal(*groups)

	return stat, p


# Functions to compute Chi-quadro

def Chi_quadro(data, feat_categorical, feat_numerical):

	contingency = pd.crosstab(data[feat_categorical], data[feat_numerical])
	stat, p, _, _ = chi2_contingency(contingency)

	return stat, p
	

#######################################################################################################################################
   
# Functions to plot
    
def plot_features_clusters(dataset, clusters, dirpath, scaler_method = 'StandardScaler'):

	'''
    Several plots:
     - dataset : the original dataset
     - clusters : label of the cluster
     - dirpath : to save    
    '''

    dataset_original = dataset.copy()
    n = len(dataset_original)
    
    # For visualization only: missing values replaced with column mean and data scaled to visualize clusters.
    columns = dataset_original.columns
    for ii in columns:
        dataset_original[ii] = pd.to_numeric(dataset_original[ii], errors = "coerce")
    dataset_original_filled = dataset_original.fillna(dataset_original.mean()).astype(float).copy()
    
    if scaler_method == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
        dataset_original_scaled = pd.DataFrame(scaler.fit_transform(dataset_original_filled), columns=dataset_original_filled.columns, 
                                               index=dataset_original_filled.index)
    elif scaler_method == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
        dataset_original_scaled = pd.DataFrame(scaler.fit_transform(dataset_original_filled), columns=dataset_original_filled.columns, 
                                               index=dataset_original_filled.index)
    else:
        dataset_original_scaled = dataset_original_filled.copy()
    
    X_tsne2 = TSNE(n_components=2,perplexity=30,random_state=42).fit_transform(np.array(dataset_original_scaled).reshape((n,-1)))
    X_tsne3 = TSNE(n_components=3,perplexity=30,random_state=42).fit_transform(np.array(dataset_original_scaled).reshape((n,-1)))

    fig1, ax = plt.subplots(1, 2)
    ax[0].scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=clusters, cmap='jet')
    fig1.delaxes(ax[1])
    ax[1] = fig1.add_subplot(1, 2, 2, projection='3d')
    ax[1].scatter(X_tsne3[:, 0], X_tsne3[:, 1],X_tsne3[:, 2], c=clusters, cmap='jet')
    plt.show()
    
    # Added the column with the identified clusters to the original datasets
    dataset_original.loc[:,'group'] = clusters

    # Show how the features behave within the identified clusters
    groups = [group for _, group in dataset_original.groupby('group')]


    for feat in columns:
    
        fig2 = plt.figure(figsize=(15, 5))
    
        # 1. Box plots
        plt.subplot(1, 3, 1)
        ax = sns.boxplot(x=feat, y = 'group', data = dataset_original, hue='group', palette= "Set1",  showmeans=True,  
                         orient = 'h', meanprops=({"marker":"*", "markerfacecolor":"blue", "markeredgecolor": "k", "markersize":10}))
        plt.title(f'Box Plot of {feat}')
        plt.xlabel(feat)
    
        # 2. Violin plots
        plt.subplot(1, 3, 2)
        ax = sns.violinplot(x=feat, y = 'group', data = dataset_original,  hue = 'group', palette= "Set1", 
                            orient='h', inner='points')
        plt.title(f'Violin Plot of {feat}')
        plt.xlabel(feat)
    
        # 3. Histogram for each cluster
        plt.subplot(1, 3, 3)
        for jj in range(len(groups)):
            sns.histplot(groups[jj][feat], alpha=0.5, label= str(jj), kde=True)
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        plt.legend()
        #fig2.savefig(dirpath + str(feat) +".png")
        plt.show()
    return 
    
    
def plot_numerical_features(dataset, dirpath):

    columns = dataset.columns

    for feat in columns:
    
        fig1 = plt.figure(figsize=(15, 5))
    
        # 1. Box plots
        plt.subplot(1, 3, 1)
        ax = sns.boxplot(x=feat, data = dataset,  showmeans=True,  
                         orient = 'h', meanprops=({"marker":"*", "markerfacecolor":"blue", "markeredgecolor": "k", "markersize":10}))
        plt.title(f'Box Plot of {feat}')
        plt.xlabel(feat)
    
        # 2. Violin plots
        plt.subplot(1, 3, 2)
        ax = sns.violinplot(x=feat, data = dataset, orient='h', inner='points')
        plt.title(f'Violin Plot of {feat}')
        plt.xlabel(feat)
    
        # 3. Histogram for each cluster
        plt.subplot(1, 3, 3)
        sns.histplot(dataset[feat], kde=True)
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        #fig1.savefig(dirpath + str(feat) +".png")
        plt.show()
    return 
    
def plot_categorical_features(dataset, dirpath):

    columns = dataset.columns

    for feat in columns:
    
        fig1 = plt.figure(figsize=(5, 5))
    
        # Count plots
        plt.subplot(1, 1, 1)
        ax = sns.countplot(dataset, x=feat)
        plt.title(f'Count Plot of {feat}')
        plt.xlabel(feat)
        #fig1.savefig(dirpath + str(feat) +".png")
        plt.show()
    return 
    
    
def plot_roc_curve(dataset, y, dirpath):

    fpr = dict()
    tpr = dict()
    roc_auc = {}
    
    columns = dataset.columns
    n = len(columns)

    for col in columns:
        fpr[col], tpr[col], _ = roc_curve(y, dataset[col])
        roc_auc[col] = auc(fpr[col], tpr[col])
        

    fig = plt.figure(figsize=(20,60))
    for ii in range(n):
        plt.subplot(int(np.ceil(n/3)), 3, ii+1)
        plt.plot(fpr[columns[ii]], tpr[columns[ii]], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(columns[ii], roc_auc[columns[ii]]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
    fig.savefig(dirpath +"ROC_curves_feature_selection.png")
    plt.show()


####################################################################################################################

# Function to perform feature selection    
    
def feature_selection(columns, X_train, y_train, dirpath_img, classification = 'binary'):
    
    # Univariate analysis: mutual information
    
    MI = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42)

    fig = plt.figure(figsize=(10,7))
    mi = pd.Series(MI)
    mi.index = columns
    mi.sort_values(ascending=False).plot.bar()
    plt.ylabel('Mutual Information')
    plt.title("Mutual information between features and target variable")
    plt.xticks(fontsize=6)
    fig.savefig(dirpath_img + "MI.png")
    plt.show()
    
    # Graphical approachs to rank features
    plot_features_clusters(X_train, y_train, dirpath_img, scaler_method = False)
    
    if classification == 'binary':
        plot_roc_curve(X_train, y_train, dirpath_img)
    
    # Multivariate analysis: Minimum Redundancy Maximum Relevance
    
    if classification == 'binary':
        mrmr_rfcq = MRMR(method="RFCQ", max_features = 20, regression=False, random_state=42)
        mrmr_mid = MRMR(method="MID", max_features = 20, regression=False, random_state=42)
        
    else:
        mrmr_rfcq = MRMR(method="RFCQ", max_features = 20, regression=False, random_state=42, scoring = 'roc_auc_ovo_weighted')
        mrmr_mid = MRMR(method="MID", max_features = 20, regression=False, random_state=42, scoring = 'roc_auc_ovo_weighted')
        
    X_rfcq = mrmr_rfcq.fit_transform(X_train, y_train)
    index_rfcq = X_rfcq.columns
    print('index RFCQ = ', np.array(index_rfcq))
    
    X_mid = mrmr_mid.fit_transform(X_train, y_train)
    index_mid = X_mid.columns
    print('index MID = ', np.array(index_mid))
        
    
    # Embedded methods: XGBoost    
    param_grid = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'max_depth': [6, 7, 8],
        'n_estimators': [40, 50, 60, 100, 150, 200],
        'subsample': [0.7, 0.8, 0.9]}

    cv_inner = StratifiedKFold( n_splits=3, shuffle=True, random_state=42)

    model = GridSearchCV( estimator = XGBClassifier(eval_metric='logloss', random_state=42), param_grid = param_grid, cv=cv_inner )
    model_fit = model.fit(X_train, y_train)

    best_model = model_fit.best_estimator_

    features_importances = best_model.feature_importances_

    feat_importances = pd.Series(features_importances, index=columns)
    
    print('Feature importance:')
    print(feat_importances)
    
    fig3 = plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 1, 1)
    ax = feat_importances.plot.bar()
    plt.ylabel('feature importance')
    plt.title("Feature importance in XGBoost")
    fig3.savefig(dirpath_img +"importances.png")
    plt.show()
    
    # Wrapper methods : recursive feature elimination
    
    # Scaler the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled_for_rfe = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    # Initialize XGBClassifier
    model = XGBClassifier(eval_metric='logloss', random_state=42)

    # Initialize RFECV
    rfe = RFE(estimator= model, n_features_to_select= 20)

    # Fit RFE
    rfe.fit(X_train_scaled_for_rfe, y_train)

    # Print the ranking
    ranking = rfe.ranking_
    print("Feature ranking RFE:")

    rank_rfe = {}
    for i, feature in enumerate(columns):
       rank_rfe[feature] = ranking[i]
    aa = sorted(((v, k) for k, v in rank_rfe.items()), reverse=False)
    for ii in range(len(columns)):
        print(aa[ii][0],' : ', aa[ii][1] )
        
    begin_loop = True
        
    while begin_loop:
        sel_features = input("Insert the selected features to build the model (separated by commas and a space). If you want all features, write all: ")
        if sel_features == 'all':
            selected_features = np.array(columns)
        else:
            selected_features = sel_features.split(', ')
        if all(item in columns for item in selected_features) == False:
            print('One or more features are not valid')
            continue
        else: 
            begin_loop = False
            
    return selected_features

#########################################################################################################################

# Function to compute clusters using Leiden algorithm

def compute_clustering(dataset_clusters, dirpath, name, scaler_method = 'StandardScaler', resolution_Leiden = 1):

	'''It applies Leiden algorithm:	
		- dataset_clusters = dataset used to identify the clusters. Missing values have already been imputed.
		- dirpath, name = name to save
		- scaler = standardization (default 'StandardScaler)
        - resolution_Leiden = resolution parameter of Leiden (default 1)
	'''


	if scaler_method == 'StandardScaler':
		scaler = preprocessing.StandardScaler()
		dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_clusters), columns=dataset_clusters.columns, index=dataset_clusters.index)
		
	elif scaler == 'RobustScaler':
		scaler = preprocessing.RobustScaler()
		dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_clusters), columns=dataset_clusters.columns, index=dataset_clusters.index)
		
	else:
		dataset_scaled = dataset_clusters.copy()

	# Compute the graph (use k-NearestNeighbors with k = 5)
	n = np.shape(dataset_scaled)[0]
	neigh = NearestNeighbors().fit(dataset_scaled)  
	W = neigh.kneighbors_graph(dataset_scaled).toarray()
	np.fill_diagonal(W, 0)
    
    
	D = (W == 1)
	boolean_graph = csr_matrix(D)

	# Apply Leiden algorithm
	if issymmetric(W):
		formula_modularity = 'newman'
	else:
		formula_modularity = 'dugue'
	leiden = Leiden(resolution = resolution_Leiden, random_state=42, modularity = formula_modularity)
	Leiden_clusters = leiden.fit_predict(boolean_graph)
	
	print(f'Resolution parameter = {resolution_Leiden}:')
	print(f'       - the associated graph is {'undirected' if issymmetric(W) else 'directed'};')
	print(f'       - with Leiden algorithm we obtain {len(set(Leiden_clusters))} clusters;')
	print(f'       - the modularity is {float(np.round(get_modularity(boolean_graph, Leiden_clusters), 6))}. \n')
    
	return Leiden_clusters