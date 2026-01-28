"""
The script performes binary classification .
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
from sklearn import preprocessing                  # used for standardization
import seaborn as sns                              # it contains tools for statistical data visualization
import matplotlib.pyplot as plt                    # it is a library for creating plots



from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

import time
import shap

import lime
import lime.lime_tabular

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score


################################################################################################################

if __name__ == '__main__':  
    
    columns = [...]
    
    dirpath_img = ....

	dirpath = "..."
    file = "..."
    sheet_name = "..."

    # Read the Excel file and convert categorical features to numeric
    dataset = pd.read_excel(dirpath + file + '.xlsx', sheet_name = sheet_name, converters={feature: functions_for_converting})
    
    dataset = dataset[columns].copy()

	target_feature = '...'
    
    X = data.drop(target_feature,axis=1)
    y = data.target_feature

    
    for ii in columns:
        X[ii] = pd.to_numeric(X[ii], errors = "coerce")
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Impute missing value in the training set with the mean of each column
    X_train = X_train.fillna(X_train.mean()).astype(float).copy()   
    
    # Scaler the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
      
    ######################################################################################################
    # 1. FEATURE SELECTION (using training set)
    
    selected_features = feature_selection(columns, X_train, y_train, dirpath_img, classification = 'multiclass')
    
    X_train_sel = X_train_scaled[selected_features].copy()
    X_test_sel = X_test_scaled[selected_features].copy()
    
    ######################################################################################################
    
    # 2. TRAIN THE MODEL
    
    # oversampling to balanced the classes
    
    smote = SMOTE(random_state=42)
    X_train_over, y_train_over= smote.fit_resample(np.array(X_train_sel), np.array(y_train))

    t = time.time()
    param_grid = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
            'max_depth': [2,3,4,5,6, 7, 8],
            'n_estimators': [40, 50, 60, 100, 150, 200],
            'subsample': [0.7, 0.8, 0.9]}

    cv_inner = StratifiedKFold( n_splits=3, shuffle=True, random_state=42)

    # fit the model
    model = GridSearchCV( estimator = XGBClassifier(eval_metric='logloss', random_state=42), param_grid = param_grid, cv=cv_inner, scoring= 'f1_macro' )
    model_fit = model.fit(X_train_over, y_train_over)
    print ('Best parameters =', model_fit.best_params_ )

    best_model = model_fit.best_estimator_
        
    ######################################################################################################
    
    # 3. EVALUATE THE MODEL
    
    y_test_pred = best_model.predict(X_test_sel)
    y_train_pred = best_model.predict(X_train_over)

    y_test_pred_proba = best_model.predict_proba(X_test_sel)
    y_train_pred_proba = best_model.predict_proba(X_train_over)

    bal_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    elapsed = time.time() - t

    print(f"Balanced accuracy: {bal_accuracy:.4f}")
    
    classes = np.unique(y_test)
    for c in classes:
        acc_c = accuracy_score(y_test[y_test == c], y_test_pred[y_test == c])
        print(f"Accuracy classe {c}: {acc_c:.2f}")
        
    print(f"Elapsed time : {elapsed:.4f}")
        
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print('TRAIN AUC :',round((roc_auc_score(y_train_over, y_train_pred_proba, multi_class='ovo', average = 'macro'))*100,2), '%')
    print('TEST AUC:',round((roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo', average = 'macro'))*100,2), '%')
    
    ######################################################################################################
    
    # 4. INTERPRETABILITY OF THE MODEL
    
    features_importances = best_model.feature_importances_

    feat_importances = pd.Series(features_importances, index=selected_features)

    result_XG = permutation_importance(best_model, X_test_sel, y_test, n_repeats=10, random_state=42, n_jobs=2)
    permutation_importances = pd.Series(result_XG.importances_mean, index=selected_features)
    
    print('\nFeature importance:')
    print(feat_importances)
    
    print('\nPermutation importance:')
    print(permutation_importances)
    
    fig5 = plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    ax = feat_importances.plot.bar()
    plt.ylabel('feature importance')
    plt.title("Feature importance in XGBoost")

    plt.subplot(1, 2, 2)
    ax = permutation_importances.plot.bar()
    plt.ylabel('permutation importance')
    plt.title("Permutation importance in Xgboost")
    fig5.savefig(dirpath_img +"model_importances.png")
    plt.show()   


    explainer_shap = shap.TreeExplainer(best_model)
    shap_values = explainer_shap.shap_values(X_test_sel)
    
    for c in classes:
        fig = shap.summary_plot(shap_values[:,:,int(c)], X_test_sel, feature_names=selected_features, max_display=10,show=False)
        plt.savefig(dirpath_img +"shap_class_" +str(c) +".png")
        plt.show()
    
    print('y_pred = ', y_test_pred)
    print('y_true = ', np.array(y_test))
    
    begin_loop = True
    
    while begin_loop:
        index = input('Insert the index of the instance you want to analyze with LIME (press Enter to exit) = ')
        
        if index == '':
            break
        else:
            index = int(index)
        print('true class = ', np.array(y_test)[index])
        print('predict class = ', y_test_pred[index])
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train_over, feature_names=selected_features,discretize_continuous=True)            
        exp = explainer_lime.explain_instance(np.array(X_test_sel)[index], best_model.predict_proba, num_features=10, top_labels = 1)
        exp.show_in_notebook(show_table=True, show_all=False)

        plt.show()
