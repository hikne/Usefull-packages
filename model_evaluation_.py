# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:33:09 2020

@author: hikne
"""


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from sklearn import metrics 
from IPython.display import display 


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
        metrics.plot_confusion_matrix(self.clf, X, y, labels=labels, normalize=normalize,cmap="Blues",xticks_rotation=90)
    
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
        y_pred=self.clf.predict(X)
        print(f'{42*"_"}    Classification report     {42*"_"}')
        print(metrics.classification_report(y,y_pred ))
        print(":::::::::: F-2 score : ",metrics.fbeta_score(y,y_pred,beta=beta,average='weighted'))
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

        ax[0].imshow(cm, cmap="Wistia")
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
                ax[0].text(j,i,str(cm[i][j]))
        ax[0].grid(False)

        #
        #-------------------------------------------#
        #            plot ROC curve                 #
        #-------------------------------------------#
        # compute probabilities
        if hasattr(self.clf,'predict_proba'):
            score = self.clf.predict_proba(X)
        else:
            score=self.clf.decision_function(X)
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
        print(f'{42*"_"} Accuracy confidence interval {42*"_"}')  
        n=len(y)
        conf_int=lambda alpha,n,e:[e-alpha*np.sqrt(e*(1-e)/n),e+alpha*np.sqrt(e*(1-e)/n)]
        ###
        confid_int=pd.DataFrame(np.array([conf_int(alp,n,metrics.accuracy_score(y,y_pred)) for alp in [1.64,1.96,2.33,2.58]]))
        confid_int.columns=['min','max']
        confid_int.index=['90%','95%','98%','99%']
        display(confid_int) 
