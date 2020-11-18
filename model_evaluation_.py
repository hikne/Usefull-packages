# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:43:09 2020

@author: hikne
"""


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from sklearn import metrics 
from IPython.display import display 
import sys
from joblib import dump
import json
import os

class model_eval:
    def __init__(self,model,class_names):
        self.clf=model 
        self.class_names=class_names 

    @staticmethod
    def quadratic_kappa(actuals, preds,w,quadratic=False):
        """
        @ Description: This function computes the Quadratic Kappa Metric,
                
        Parameters
        ----------
        actual      - Required  : true labels  (1D Array/List like)
        preds       - Required  : predicted labels (1D Array/List like)
        w           - Required  : error weight matrix (2D array  `dim=[N*N] with N: n_classes`)
        quadratic   - Optional  : square w or not  (Boolean  ~ default=True)

        Returns
        -------
        [Float] Quadratic Weighted Kappa metric score between the actual and the predicted values.
        
        """    
        N=w.shape[0]
        if quadratic:
            w=w*w
        O = metrics.confusion_matrix(actuals, preds)
        
        act_hist,pred_hist=np.zeros([N]),np.zeros([N])
        for item in actuals: 
            act_hist[item]+=1   
        for item in preds: 
            pred_hist[item]+=1
                            
        E = np.outer(act_hist, pred_hist)
        E ,O = E/E.sum(), O/O.sum()
        num,den=0,0
        for i in range(len(w)):
            for j in range(len(w)):
                num+=w[i][j]*O[i][j]
                den+=w[i][j]*E[i][j]
        return (1 - (num/den)) 
    
    def plot_confMatrix(self,X,y,labels,normalize=None):
        """
       @ Description: plot ROC curve,
                
        Parameters
        ----------
            X             - Required  : Input samples (array-like or sparse matrix of shape = [n_samples, n_features])
            y             - Required  : True binary labels (1D array, shape = [n_samples])
            labels        - Optional  : labels names (List)
            normmalize    - Optional  : normalization axis {'pred': predicted labels, 'true': true labels, 'all':both} (Str)
        """
        plt.style.use('default')
        np.set_printoptions(precision=2)
        metrics.plot_confusion_matrix(self.clf, X, y, labels=labels, normalize=normalize,
                                      cmap=plt.cm.Blues,xticks_rotation=90)
    
    def roc_curve(self,X,y):
        """
        @ Description: plot ROC curve,
                
        Parameters
        ----------
        X               - Required  : Input samples (array-like or sparse matrix of shape = [n_samples, n_features])
        y               - Required  : True binary labels (1D array, shape = [n_samples])
        
        """  
        # split data
        plt.style.use('default')
        plt.figure(figsize=(8,4))
        ax = plt.gca()
        metrics.plot_roc_curve(self.clf, X,y,ax=ax)
        plt.show()

    def model_perf(self,X,y,norm=None,beta=1,kappa=False,w=None,roc_curve=True,
                            proba_chart=True,positive_class=None,**kwargs):
        """
        @ Description: This function performs a global classification model performance evaluation with several insights:
            - classification report
            - fbeta score
            - QWE kappa score
            - confusion matrix
            - roc curve (binary classifiaction)
            - probabilty distribution chart
            - confidence interval

        Parameters
        ----------
        X               - Required  : Input samples (array-like or sparse matrix of shape = [n_samples, n_features])
        y               - Required  : True binary labels (1D array, shape = [n_samples])
        norm            - Optional  : Normalizing confusion matrix (Str ~ default=None `must be in None, 'true' or 'pred'`)
        beta            - Optional  : fbeta weight of precision in harmonic mean (float ~ default=1).
        kappa           - Optional  : Whether or not to compute kappa score (Boolean ~ default=False)
        w               - Optional  : kappa error weight matrix (2D array  `dim=[N*N] with N: n_classes`) If kappa=True then w must be given.
        roc_curve       - Optional  : Whether or not to plot ROC curve (Boolean ~ default=True `valid only for binary classification`)
        proba_chart     - Optional  : Whether or not to plot probabilty distribution chart (Boolean ~ default=True)
        positive_class  - Optiona   : positive class for which probability chart will be displayed (Integer ~ default=None `i.e maximum value in y_test will be taken`)



        Others Parameters
        -----------------
        **kwargs
        quadratic: kappa
        average: fbeta
        """
        #-------------------------------------------#
        #      print classification report          #
        #-------------------------------------------#
        plt.style.use('default')
        
        y_pred=self.clf.predict(X)
        
        print(f'{42*"_"}    Classification report     {42*"_"}')
        print(metrics.classification_report(y,y_pred ))
        
        fbscore=metrics.fbeta_score(y,y_pred,beta,average='weighted')
        print(f":::::::::: F-{beta} score : {fbscore}")
        
        # compute kappa
        if kappa:
            qk=self.quadratic_kappa(y, y_pred,w)
            print(':::::::::: QWE= ',qk)
            
        # show confusion matrix, ROC curve and probability chart
        print(f'{42*"_"}         Visual report        {42*"_"}') 
        n_charts=1+roc_curve+proba_chart
        #
        _, ax = plt.subplots(1, n_charts, figsize=(5*n_charts,4))
        #-------------------------------------------#
        #          plot confusion matrix            #
        #-------------------------------------------#
        cm = metrics.confusion_matrix(y,y_pred)
        # if asked normalize confusion matrix
        if norm!=None:
            normax=['pred','true']
            cm = np.round(100*cm.astype('float') / cm.sum(axis=normax.index(norm))[:, np.newaxis],2)

        ax[0].imshow(cm, cmap=plt.cm.Wistia)
        ax[0].set_title('confusion matrix')
        ax[0].set_ylabel('True label')
        ax[0].set_xlabel('Predicted label')
        tick_marks = np.arange(len(self.class_names))
        ax[0].set_xticks(tick_marks)
        ax[0].set_xticklabels(self.class_names, rotation=45)
        ax[0].set_yticks(tick_marks)
        ax[0].set_yticklabels(self.class_names)
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                ax[0].text(j-.2,i,str(cm[i][j]))
        ax[0].grid(False)

        #
        #-------------------------------------------#
        #            plot ROC curve                 #
        #-------------------------------------------#
        # compute probabilities
        if hasattr(self.clf,'predict_proba'):
            score = self.clf.predict_proba(X)
        else:
            score=clf.decision_function(X)
            
        # check if it's a binary classification
        if roc_curve:
            assert len(np.unique(y))==2, ('ROC curve is valid only for binary classification')
            fpr, tpr, _ = metrics.roc_curve(y, score[:, 1])
            ax[1].plot([0, 1], [0, 1], 'k--')
            aucf = metrics.auc(fpr, tpr)
            ax[1].plot(fpr, tpr, label='auc=%1.5f' % aucf)
            ax[1].set_title('ROC-curve')
            ax[1].set_xlabel("P(FP)")
            ax[1].set_ylabel("P(TP)")
            ax[1].legend()
            
        #-------------------------------------------#
        #    plot probability distribution chart    #
        #-------------------------------------------#
        if proba_chart:
            if positive_class==None:
                positive_class=int(max(y))
            # we assume that ROC curve is displayed -> probability chart is the 3rd chart (i.e subplot index==2)
            chart_idx=2
            
            # if ROC is not requested then proability chart index ==1
            if n_charts!=3:
                chart_idx=1
            ax[chart_idx].hist(score[y!=positive_class,positive_class],bins=50, label='Negative class',alpha=.7, color='blue',density=True)
            ax[chart_idx].hist(score[y==positive_class,positive_class],bins=50, label='Postive class',alpha=.7 ,color='orange',density=True)
            ax[chart_idx].set_title('postive class probabilities distribution')
            
            # display .5 boundary
            ymax=max(max(np.histogram(score[y!=positive_class,positive_class],bins=50,density=True)[0]),
                    max(np.histogram(score[y==positive_class,positive_class],bins=50,density=True)[0]))+1
            ax[chart_idx].plot([0.5, 0.5], [0, ymax], 'g--')
            ax[chart_idx].text(0.4,ymax/2,'Boundary',fontsize=12,rotation=45)
            ax[chart_idx].legend()
        plt.show()

        #-------------------------------------------#
        #             Additional metrics            #
        #-------------------------------------------#
        print(f'{42*"_"} F{beta} score confidence interval {42*"_"}')  
        n=len(y)
        conf_int=lambda alpha,n,e:[e-alpha*np.sqrt(e*(1-e)/n),e+alpha*np.sqrt(e*(1-e)/n)]
        ###
        confid_int=pd.DataFrame(np.array([conf_int(alp,n,fbscore) for alp in [1.64,1.96,2.33,2.58]]))
        confid_int.columns=['min','max']
        confid_int.index=['90%','95%','98%','99%']
        display(confid_int) 

    def percentage_by_range(self,X,y,classIdx):
        # probabilité de la classe 0 (NO MOVER / SLOW MOVER)
        if hasattr(self.clf,'predict_proba'):
            score = self.clf.predict_proba(X)[:,classIdx]
        else:
            score=clf.decision_function(X)[:,classIdx]
            
        #
        res=pd.DataFrame({'score':score,'true label':y})
        # score range
        res['range']=pd.cut(res['score'],bins=np.linspace(0,1,11)).astype('object')
        
        ## effectif par classe en fonction du range de score
        out=pd.crosstab(res['range'],res['true label']).reset_index()
                
        # normaliser les chiffres
        out.rename(columns={i:self.class_names[i] for i in range(len(self.class_names))},inplace=True)
        #
        out['Total']=out[self.class_names].sum(axis=1)
        out['Total']=round(100*out['Total']/out['Total'].sum(),2)
        #
        out[self.class_names]=out[self.class_names].apply(lambda x:round(100*x/x.sum(),2),axis=1)
        #
        out.index=out['range']
        out.drop(columns=['range'],inplace=True)
        #
        display(out.style.format({col: "{:20,.0f}%" for col in out.columns}).set_properties(**{'color': 'white'}
                                                      ,subset=[self.class_names[classIdx]]).background_gradient(cmap='brg',subset=[self.class_names[classIdx]]))
        #return out[['range']+self.class_names+['(%) tot']].reset_index(drop=True)
    
        
    def feature_influence(self,features,nb=-1,figsize=(12,10)):
        """
        @ Description: This function displyas 'nb' most influent in the model decision.
        @ Input:
                +> features: [List] feature list.
                +> nb: [Integer] number of features to display (default=-1, i.e display all).
        @ Output: [Figure]
        """
        if hasattr(self.clf,'feature_importances_'):
            feature_imp=self.clf.feature_importances_
        elif hasattr(self.clf,'coef_'):
            feature_imp=self.clf.coef_
        else:
            sys.exit()
            
        # reshape feature importances
        feature_imp=feature_imp.reshape(-1,)
        
        imp_=pd.DataFrame({'feature':features,'importance':feature_imp},index=range(len(features))).sort_values(by=['importance'],ascending=True)
        if nb==-1:
            nb=20
            
        imp_.iloc[:nb,:].plot.barh(x='feature',y='importance',color='lightblue',edgecolor= 'blue',linestyle='-',figsize=figsize)
        plt.show()
    
    def save_model(self,features,dir_path,model_name):
        # save joblib file
        dump(model,os.path.join(dir_path,f'{model_name}.joblib'))
        # save feature importance if available, Otherwise just feature names
        if hasattr(self.model,'feature_importances_'):
            #
            #
            with open(os.path.join(dir_path,f'{model_name}_feature_imp.txt'), 'w') as outfile:
                json.dump(feat_imp, dict(zip(features,[str(f) for f in self.model.feature_importances_])))

        print('done!')

