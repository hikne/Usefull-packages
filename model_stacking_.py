##################
##  Libririaes 
##################
import numpy as np
from sklearn import model_selection 
from tqdm import tqdm
import warnings
from scipy.special import expit,softmax
import copy
import warnings
from collections import defaultdict
import platform
import inspect
import re


class model_fusion:
    def __init__(self,estimators,kfold,scoring='neg_log_loss',estimator_names=None,activation='softmax',seed=42):
        ## if estimator names not given => set clf <i> as estimator names
        if estimator_names==None:
            estimator_names=[f'clf <{i}>' for i in range(len(estimators))]
        self.estimators={name:clf for name,clf in zip(estimator_names,estimators)}
        self.kfold=kfold
        self.scoring=scoring
        self.seed=seed
        self.activation=activation
             
    def get_params(self):
        return {'estimators':{key:clf.get_params() for key,clf in self.estimators.items()},
                'kfold':self.kfold,
                'scoring':self.scoring,
                'seed':self.seed,
                'activation':self.activation,
                'estimator_weights':self.weights
               } 
    
    def set_params(self,**params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        valid_params = self.get_params()
        nested_params = defaultdict(dict)  # grouped by prefix
        
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
    
    def compute_weights(self,X,y):
        print("**  Running task: estimator weight computation ...  **")
        self.weights={}
        ## compute cross-val score mean for each estimator ==> estimator weight
        for key,clf in tqdm(self.estimators.items()):
            self.weights[key]=model_selection.cross_validate(clf, X,y, cv=self.kfold, scoring=self.scoring,n_jobs=-1,return_estimator=False)['test_score'].mean()
        # if a negative values error is used for the scoring
        np.array(self.weights.values())
            
    def compute_feature_importances(self):
        res=[]
        
        # for each base-estimator get feature_importances then weight it with it's estimator-weight
        for key,clf in self.estimators.items():
            # check if model has coef_ property (ex: linear model)
            if hasattr(clf,'coef_'):
                # wieght all values with clf weight
                res.append(self.weights[key]*clf.coef_)
            else:
                res.append(self.weights[key]*clf.feature_importances_)
        res=np.array(res).sum(axis=0)
        self.feature_importances=res/res.sum()
    
    def fit(self,X,y):
        # compute weights
        self.compute_weights(X,y)
        # number of classes
        self.n_class=len(np.unique(y))
        print("**  Running task: fitting estimators...  **")
        for clf in tqdm(self.estimators.values()):
            clf.fit(X,y)
            
        self.compute_feature_importances()
            
            
    
    
    def normalize_proba(self,p):
        if self.activation=='sigmoid':
            p=expit(p)
            return p.astype('float') / p.sum(axis=1)[:, np.newaxis]
        elif self.activation not in ('softmax','sigmoid'):
            warnings.warn("Warning........... unkown activation function! \n activation function must be either  'softmax' or 'sigmoid' \n Note that softmax is used!")
        return softmax(p,axis=1)
        

    def predict_proba(self,X):
        p=np.zeros((X.shape[0],self.n_class))
        for key in self.estimators.keys():
            p+=self.weights[key]*self.estimators[key].predict_proba(X)
        return self.normalize_proba(p)

    def predict(self,X):
        p=self.predict_proba(X)
        return np.argmax(p,axis=1)
    
    



