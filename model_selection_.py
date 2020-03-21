# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:51:43 2020

@author: hikne
"""
##################################
##          libraries           ##
##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
## ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,SGDClassifier,SGDRegressor,BayesianRidge,LinearRegression,Lars
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingClassifier,AdaBoostRegressor
from sklearn import svm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier,XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
# more Classifiers
from sklearn.ensemble import  BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import ExtraTreeClassifier
## save the model
from joblib import dump
import json



class modelSelection:
    """
    @ model selection with model benchmark and hyper-parameter tunning 
    """
    def __init__(self,features,target,task='classification',score='accuracy',test_size=.2,cv=5,random_state=42):
        """
        Define a modelSelection object.
        Parameters
        ----------
            features          - Required  : feature names (Array-like)
            target            - Required  : target name (String)
            task              - Optional  : task kind (String ~ default=`classification` [value in `classification`,`regression`])
            score             - Optional  : scoring method (Str  ~ default='accuracy')
            test_size         - Optional  : test set size (Float ~ default=0.2)
            cv                - Optional  : cross-validation number of folds (Int ~ default=5)
            random_state      - Optional  : random-state parameter (Int ~ default=42)
            
        """
        self.features=features
        self.target=target
        self.task=task
        self.score=score
        self.test_size=test_size
        self.cv=cv
        self.random_state=random_state
    
    
    #--------------------------------------------------------------------------
    def multiple_models_simulation(self,data):
        """
        @ Description: Compute cross-validation score for different kind of estimators.
                
        Parameters
        ----------
        data        - Required  : dataset (DataFrame)
       
        
        """    
        
        X_train, _, y_train, _ = train_test_split(data[self.features], data[self.target], 
                                                         test_size=self.test_size, random_state=self.random_state)
        
        
        ############################
        # Construct some pipelines #
        ############################
        # linear models
        if self.task=='classification':
            Pipe_mod ={}
            Pipe_mod['Logistic Regression'] = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty='l2',random_state=42,n_jobs=-1))])
            Pipe_mod['LDA'] = Pipeline([('scl', StandardScaler()),('clf', LinearDiscriminantAnalysis())])
            Pipe_mod['QDA']=  Pipeline([('scl', StandardScaler()),('clf', QuadraticDiscriminantAnalysis())])  
            Pipe_mod['SGD'] = Pipeline([('scl', StandardScaler()),('clf',SGDClassifier(random_state=42))])
            
            # tree based models
            Pipe_mod['Random Forest '] = Pipeline([('clf', RandomForestClassifier(n_estimators=400,max_depth=15,random_state=42))])
            Pipe_mod['AdaBoost'] = Pipeline([('clf', AdaBoostClassifier(n_estimators=250,learning_rate=.01,random_state=42))])				
            Pipe_mod['XGB']= Pipeline([('clf', XGBClassifier(n_estimators=150,max_depth=15,random_state=42))])
            ## gardient boosted
            
            Pipe_mod['GB']= Pipeline([('clf', GradientBoostingClassifier(n_estimators=250,max_depth=7,random_state=42))])
            # generative models
            Pipe_mod['GNB']= Pipeline([('clf', GaussianNB())])
            
            # kernel models
            Pipe_mod['SVM'] = Pipeline([('scl', StandardScaler()),('clf', svm.SVC(random_state=42))])
                            
            # knn
            Pipe_mod['KNN'] = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors=7))])
                            
            # NN model
            Pipe_mod['MLP'] = Pipeline([('scl', StandardScaler()),
                            ('clf', MLPClassifier(hidden_layer_sizes=(128,64,128,data[self.target].nunique()),activation='relu',batch_size=64,random_state=42))])
        	
        elif self.task=='regression':
            Pipe_mod ={}
            Pipe_mod['Bayesian Regression'] = Pipeline([('scl', StandardScaler()),('clf', BayesianRidge())])
            Pipe_mod['Linear Regression'] = Pipeline([('clf', LinearRegression(fit_intercept=True))])
            Pipe_mod['Lars']=  Pipeline([('scl', StandardScaler()),('clf', Lars(n_nonzero_coefs=1))])  
            Pipe_mod['SGD'] = Pipeline([('scl', StandardScaler()),('clf',SGDRegressor(max_iter=1000, tol=1e-3))])
            # tree based models
            Pipe_mod['Random Forest '] = Pipeline([('clf', RandomForestRegressor(n_estimators=400,max_depth=15,random_state=42))])
            Pipe_mod['AdaBoost'] = Pipeline([('clf', AdaBoostRegressor(n_estimators=250,learning_rate=.001,random_state=42))])				
            Pipe_mod['XGB']= Pipeline([('clf', XGBRegressor(n_estimators=150,max_depth=15,random_state=42,objective='reg:squarederror'))])
            ## gardient boosted
            
            Pipe_mod['GB']= Pipeline([('clf', GradientBoostingRegressor(n_estimators=250,max_depth=7,random_state=42))])
            # generative models
            Pipe_mod['GNB']= Pipeline([('clf', GaussianNB())])
            # kernel models
            Pipe_mod['SVM'] = Pipeline([('scl', StandardScaler()),('clf', svm.SVR())])
            # knn
            Pipe_mod['KNN'] = Pipeline([('clf', KNeighborsRegressor(n_neighbors=7))])
            # NN model
            Pipe_mod['MLP'] = Pipeline([('scl', StandardScaler()),
                            ('clf', MLPRegressor(hidden_layer_sizes=(128,64,128,25),activation='relu',batch_size=64,random_state=42))])
        	
            
        ############################		
        # compute cross_val scores #
        ############################
        kfold = model_selection.KFold(n_splits=self.cv, random_state=self.random_state)
        results = []
        names = []
        X,y=data[self.features].copy(),data[self.target].copy()
        for name, model in Pipe_mod.items():
            cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=self.score)
            results.append(cv_results)
            names.append(name)
            msg = "%s %s : %f (+/- %f)" % (name,self.score, cv_results.mean(), cv_results.std())
            print(msg)
        
        	## boxplot algorithm comparison
        plt.figure(figsize=(8,5))
        chart=sns.boxplot(x=names,y=results)
        plt.title('Algorithm benchmark')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.show()
        
        
    #--------------------------------------------------------------------------   
    def model_tunnig(self,data,estimators):
        """
        @Description: make a model selection with a hyper-parameter tunning on a given list of models.
        Parameters
        ----------
            data              - Required  : dataset (DataFrame)
            estimators        - Required  : sample of model names among:
                                |+ `LDA` for `LinearDiscriminantAnalysis`
                                |+ `QDA` for `QuadraticDiscriminantAnalysis`
                                |+ `AdaBoost` for `AdaBoostClassifier`
                                |+ `Bagging` for `BaggingClassifier`
                                |+ `Extra Trees Ensemble` for `ExtraTreesClassifier`
                                |+ `Gradient Boosting` for `GradientBoostingClassifier`
                                |+ `Random Forest` for `RandomForestClassifier`
                                |+ `Ridge` for `RidgeClassifier`
                                |+ `SGD` for `SGDClassifier`
                                |+ `BNB` for `BernoulliNB`
                                |+ `GNB` for `GaussianNB`
                                |+ `KNN` for KNeighborsClassifier
                                |+ `MLP` for MLPClassifier
                                |+ `LSVC` for LinearSVC
                                |+ `NuSVC` for NuSVC
                                |+ `SVC` for SVC
                                |+ `DTC` for DecisionTreeClassifier
                                |+ `ETC` for ExtraTreeClassifier
                                |+ `XGBOOST` for XGBClassifier
        """
        X_train, X_test, y_train, y_test = train_test_split(data[self.features], data[self.target], 
                                                         test_size=self.test_size, random_state=self.random_state)
        #----------------------------------#
        #           Classifiers            #
        #----------------------------------#
        # Create list of tuples with classifier label and classifier object 
        classifiers = {}
        classifiers.update({"LDA": LinearDiscriminantAnalysis()})
        classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
        classifiers.update({"AdaBoost": AdaBoostClassifier()})
        classifiers.update({"Bagging": BaggingClassifier()})
        classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
        classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
        classifiers.update({"Random Forest": RandomForestClassifier()})
        classifiers.update({"Ridge": RidgeClassifier()})
        classifiers.update({"SGD": SGDClassifier()})
        classifiers.update({"BNB": BernoulliNB()})
        classifiers.update({"GNB": GaussianNB()})
        classifiers.update({"KNN": KNeighborsClassifier()})
        classifiers.update({"MLP": MLPClassifier()})
        classifiers.update({"LSVC": LinearSVC()})
        classifiers.update({"NuSVC": NuSVC()})
        classifiers.update({"SVC": SVC()})
        classifiers.update({"DTC": DecisionTreeClassifier()})
        classifiers.update({"ETC": ExtraTreeClassifier()})
        classifiers.update({"XGBOOST": XGBClassifier()})
        
        # Create dict of decision function labels
        DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}
        # Create dict for classifiers with feature_importances_ attribute
        #FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}
        #
        # Initiate parameter grid

        parameters = {}
        
        # Update dict with LDA
        parameters.update({"LDA": {"classifier__solver": ["svd"], 
                                                 }})
        #----------------------------------#
        #       Hyper-parameters           #
        #----------------------------------#
        # Update dict with QDA
        parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)], 
                                                 }})
        # Update dict with AdaBoost
        parameters.update({"AdaBoost": { 
                                        "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                        "classifier__n_estimators": [200],
                                        "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
                                         }})
        
        # Update dict with Bagging
        parameters.update({"Bagging": { 
                                        "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                        "classifier__n_estimators": [200],
                                        "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                        "classifier__n_jobs": [-1]
                                        }})
        
        # Update dict with Gradient Boosting
        parameters.update({"Gradient Boosting": { 
                                                "classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                                "classifier__n_estimators": [200],
                                                "classifier__max_depth": [2,3,4,5,6],
                                                "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                                "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                                "classifier__max_features": ["auto", "sqrt", "log2"],
                                                "classifier__subsample": [0.8, 0.9, 1]
                                                 }})
        
        
        # Update dict with Extra Trees
        parameters.update({"Extra Trees Ensemble": { 
                                                    "classifier__n_estimators": [200],
                                                    "classifier__class_weight": [None, "balanced"],
                                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                                    "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                                    "classifier__criterion" :["gini", "entropy"]     ,
                                                    "classifier__n_jobs": [-1]
                                                     }})
        
        
        # Update dict with Random Forest Parameters
        parameters.update({"Random Forest": { 
                                            "classifier__n_estimators": [200],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"]     ,
                                            "classifier__n_jobs": [-1]
                                             }})
        
        # Update dict with Ridge
        parameters.update({"Ridge": { 
                                    "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                                     }})
        
        # Update dict with SGD Classifier
        parameters.update({"SGD": { 
                                    "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                                    "classifier__penalty": ["l1", "l2"],
                                    "classifier__n_jobs": [-1]
                                     }})
        
        
        # Update dict with BernoulliNB Classifier
        parameters.update({"BNB": { 
                                    "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                                     }})
        
        # Update dict with GaussianNB Classifier
        parameters.update({"GNB": { 
                                    "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
                                     }})
        
        # Update dict with K Nearest Neighbors Classifier
        parameters.update({"KNN": { 
                                    "classifier__n_neighbors": list(range(1,31)),
                                    "classifier__p": [1, 2, 3, 4, 5],
                                    "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                    "classifier__n_jobs": [-1]
                                     }})
        
        # Update dict with MLPClassifier
        parameters.update({"MLP": { 
                                    "classifier__hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                                    "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                                    "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                                    "classifier__max_iter": [100, 200, 300, 500, 1000, 2000],
                                    "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
                                     }})
        
        parameters.update({"LSVC": { 
                                    "classifier__penalty": ["l2"],
                                    "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
                                     }})
        
        parameters.update({"NuSVC": { 
                                    "classifier__nu": [0.25, 0.50, 0.75],
                                    "classifier__kernel": ["linear", "rbf", "poly"],
                                    "classifier__degree": [1,2,3,4,5,6],
                                     }})
        
        parameters.update({"SVC": { 
                                    "classifier__kernel": ["linear", "rbf", "poly"],
                                    "classifier__gamma": ["auto"],
                                    "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
                                    "classifier__degree": [1, 2, 3, 4, 5, 6]
                                     }})
        
        
        # Update dict with Decision Tree Classifier
        parameters.update({"DTC": { 
                                    "classifier__criterion" :["gini", "entropy"],
                                    "classifier__splitter": ["best", "random"],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                     }})
        
        # Update dict with Extra Tree Classifier
        parameters.update({"ETC": { 
                                    "classifier__criterion" :["gini", "entropy"],
                                    "classifier__splitter": ["best", "random"],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                     }})
        
        
        parameters.update({"XGBOOST": { 
                                    "classifier__max_depth" :[3,5,7,10,15,25,35],
                                    "classifier__n_estimators": [50,70,120,200],
                                    "classifier__reg_lambda": [1e-3,.1,1,10,1e2],
                                    "classifier__learning_rate" : [1e-2,1e-1,.2,.5,.7],
                                     }})

        # Initialize dictionary to store results
        results = {}
        
        # Tune and evaluate classifiers
        for classifier_label in estimators:
            # Print message to user
            print(f"Now tuning {classifier_label}.")
        
            # Scale features via Z-score normalization
            scaler = StandardScaler()
        
            # Define steps in pipeline
            clf=classifiers[classifier_label]
            steps = [("scaler", scaler), ("classifier",clf )]
        
            # Initialize Pipeline object
            pipeline = Pipeline(steps = steps)
        
            # Define parameter grid
            param_grid = parameters[classifier_label]
        
            # Initialize GridSearch object
            gscv = model_selection.GridSearchCV(pipeline, param_grid, cv =self.cv,  n_jobs= -1, verbose = 1, scoring = self.score)
        
            # Fit gscv
            gscv.fit(X_train, np.ravel(y_train))  
        
            # Get best parameters and score
            best_params = gscv.best_params_
            best_score = gscv.best_score_
        
            # Update classifier parameters and define new pipeline with tuned classifier
            tuned_params = {item[12:]: best_params[item] for item in best_params}
            clf.set_params(**tuned_params)
        
            # Make predictions
            if classifier_label in DECISION_FUNCTIONS:
                y_pred = gscv.decision_function(X_test)
            else:
                y_pred = gscv.predict_proba(X_test)[:,1]
        
            # Evaluate model
            auc = metrics.roc_auc_score(y_test, y_pred)
        
            # Save results
            result = {"Classifier": gscv,
                      "Best Parameters": best_params,
                      "Training AUC": best_score,
                      "Test AUC": auc}
        
            results.update({classifier_label: result})
        self.results=results   
        
    
    #--------------------------------------------------------------------------
    def show_result(self,data,estimators,show=True):
        
        # Initialize auc_score dictionary
        auc_scores = {
                      "Classifier": [],
                      "AUC": [],
                      "AUC Type": []
                      }
        
        # Get AUC scores into dictionary
        assert hasattr(self,'results'),("self doesn't have any attribute results! Please call `model_tunnig(self,data,estimators)`  before !")
        results=self.results
        for classifier_label in results:
            auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                               "AUC": [results[classifier_label]["Training AUC"]] + auc_scores["AUC"],
                               "AUC Type": ["Training"] + auc_scores["AUC Type"]})
        
            auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                               "AUC": [results[classifier_label]["Test AUC"]] + auc_scores["AUC"],
                               "AUC Type": ["Test"] + auc_scores["AUC Type"]})
        
        # Dictionary to PandasDataFrame
        auc_scores = pd.DataFrame(auc_scores)
        
        # Set graph style
        sns.set(font_scale = 1.75)
        sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                       "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                       'ytick.color': '0.4'})
        
        
        # Colors
        training_color = sns.color_palette("RdYlBu", 10)[1]
        test_color = sns.color_palette("RdYlBu", 10)[-2]
        colors = [training_color, test_color]
        
        # Set figure size and create barplot
        f, ax = plt.subplots(figsize=(12, 9))
        
        sns.barplot(x="AUC", y="Classifier", hue="AUC Type", palette = colors,
                    data=auc_scores)
        
        # Generate a bolded horizontal line at y = 0
        ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)
        
        # Turn frame off
        ax.set_frame_on(False)
        
        # Tight layout
        plt.tight_layout()
        plt.show()
        if show:
            display(auc_scores.sort_values(by='AUC'))
        # Save Figure
        #plt.savefig("AUC Scores.png", dpi = 1080)
        
    #--------------------------------------------------------------------------
    def save_model(self,model,DirPath,modName):
        ## save model
        dump(model, f'{DirPath}{modName}.joblib')
        ## save feature importances
        feat_imp={self.features[i]:str(model.feature_importances_[i]) for i in range(len(self.features))}
        with open(f'{DirPath}{modName}_feature_imp.txt', 'w') as outfile:
            json.dump(feat_imp, outfile)
        print('done!')
                
                
                
