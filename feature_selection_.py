##################################
##          libraries           ##
##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random as rd
## ML libraries
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn import model_selection


class FeatureSelector:
    def __init__(self,estimator,nbmax_feat,nbIter=10**3):
        """
        Call in a loop to create terminal progress bar
        @params:
            estimator   - Required  : main estimator (classifier/Regressor) used to compute scores/losses (estimator)
            nbmax_feat  - Required  : Maximum number of features to consider in every iteration (Int)
            nbIter      - Optional  : Loop size/Number of iterations (Int)
        """
        self.model=estimator
        self.nbmax_feat=nbmax_feat
        self.nbIter=nbIter

    @staticmethod
    def printProgressBar (iteration, total, prefix = '', suffix = ''):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
        """
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(30 * iteration // total)
        bar = '█' * filledLength + '-' * (30 - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '')
        # Print New Line on Complete
        if iteration == total: 
            print()

    def optimalSize(self,data,features,target='label',loopsize=50,score='accuracy',cv=5,seed=42):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
        """
        #-----------------------------------#
        #  compute feature's sample scores  #
        #-----------------------------------#
        ## initialize Kfolds for cross-validation
        kfold = model_selection.KFold(n_splits=cv, random_state=seed)
        ## compute model scores for each randomdly drawn sample of features
        test_score=[]
        for k in range(2,self.nbmax_feat+1):
            k_scores=[]
            for iteration in range(loopsize):
                ## draw a features subsample 
                sub_feat=rd.choice(a=features,size=k,replace=False)
                ## compute cross validation score mean
                cross_val= model_selection.cross_validate(self.model, data[sub_feat],data[target], cv=kfold, scoring=score,n_jobs=-1,return_estimator=False)
                k_scores.append(cross_val['test_score'].mean())
                self.printProgressBar (loopsize*(k-2)+iteration+1, (self.nbmax_feat-1)*loopsize, prefix = 'job running: optimal size ')
            test_score.append(k_scores)
        # compute k-scores mean & std    
        test_score=np.array(test_score)
        test_score_mean=test_score.mean(axis=1)
        test_score_std=test_score.std(axis=1)
        #-----------------------------------#
        #    plot feature's sample scores   #
        #-----------------------------------#
        plt.style.use('default')
        plt.figure(figsize=(8,4))
        # plot scores mean
        plt.plot(range(2,self.nbmax_feat+1),test_score_mean,'k--')
        # plot scores profile
        plt.fill_between(x=range(2,self.nbmax_feat+1),y1=test_score_mean-test_score_std,
                    y2=test_score_mean+test_score_std,alpha=.2)
        # plot score's max
        idx=np.argmax(test_score_mean)
        plt.plot(idx,test_score_mean[idx],'*',color='orange')
        # figure limits
        plt.xlim(left=1,right=self.nbmax_feat+1)
        plt.ylim(bottom=min(test_score_mean-test_score_std)-.3,top=max(test_score_mean+test_score_std)+.3)
        # labels
        plt.xlabel('features sample size')
        plt.ylabel('score')
        plt.title('optimal features sample size')
        plt.show()
        

    def FDB_loss(self,X,y,class_weight=None):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
        """
        # check dimension consistency
        try: 
            X.shape[0] == y.shape[0]
        except:
            print("Incosistant dimensions X.shape[0] != y.shape[0].")
        # make predictions in order to compute loss
        y_pred=self.model.predict(X)
        # unique classes
        C=np.sort(np.unique(y))
        # if class_weight is not given, then set equal weights
        if class_weight==None:
            class_weight=np.ones(len(C))
        else:
            class_weight=np.array(sorted(class_weight.items()))[:,1]
        # Initilize features loss with 0.0
        L=np.zeros((C.shape[0],X.shape[1]))
        for i,c in enumerate(C,start=0):
            # get good & bad predictions
            Good,Bad=X[(y==y_pred) & (y==c)].T.values,X[(y!=y_pred) & (y==c)].T.values
            for g,b,j in zip(Good,Bad,range(L.shape[1])):
                L[i,j]=1-ks_2samp(g,b).statistic
        return class_weight.dot(L)


    def evolutive_proba_app(self,data,features,target='label',test_size=.2,class_weight=None,seed=42):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
        """
        
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], 
                                                     test_size=test_size, random_state=seed)
        # initialize uniform probabilty
        p=np.ones(len(features))/len(features)
        for iteration in range(self.nbIter):
            # draw a features sample
            sub_features=rd.choice(a=features,p=p,replace=False,size=self.nbmax_feat)
            # train estimator in order to compute feature-class loss
            self.model.fit(X_train[sub_features],y_train)
            # compute features sample loss
            sub_feat_loss=self.FDB_loss(X_test[sub_features],y_test,class_weight)
            # for all features -> if feature was used take computed loss, else: take 1 (i.e: keep same probaility)
            loss=np.ones(len(features))
            for i,col in enumerate(sub_features):
                loss[list(features).index(col)]=sub_feat_loss[i]
            # update features probabilities -> with features loss
            print(sub_features,sub_feat_loss)
            p=loss*p
            p/=p.sum()
            self.printProgressBar (iteration+1, self.nbIter, prefix = 'job running: evolutive probability feature selection process ')
        #
        p=pd.DataFrame({'feature':features,'importance':p}).sort_values(by='importance',ascending=False)
        # make a plot
        plt.figure(figsize=(12,5))
        plt.bar(x='feature',height='importance',data=p)
        plt.show()
        return p

    def feature_imp_app(self,data,features,target,cv=7,score='accuracy',score_pen_tresh=None):
        """
        Operate a classifier based feature selection.
        @params:
            data              - Required  : dataset (DataFrame)
            features          - Required  : features list (List)
            cv                - Optional  : cross-validation number of folds (Int) [default='accuracy']
            score             - Optional  : scoring method (Str)
            score_pen_tresh   - Optional  : score threshold of penality (Float) [default=None]
        """
        ## initialize Kfolds for cross-validation
        kfold = model_selection.KFold(n_splits=cv, random_state=42)
        ## compute model scores for each randomdly drawn sample of features
        # set feature scores to 0.0
        scores={col:0.0 for col in features}
        
        for i in range(self.nbIter):
            ## draw a features subsample 
            sub_feat = np.random.choice(a=features,size=np.random.randint(low=5,high=self.nbmax_feat),replace=False)
            ## compute cross validation score mean
            cross_val = model_selection.cross_validate(self.model, data[sub_feat],data[target], cv=kfold, scoring=score,n_jobs=-1,return_estimator=True)
            #
            sco = cross_val['test_score'].mean()
            #
            feature_importances = np.array([estimator.feature_importances_ for estimator in cross_val['estimator']]).mean(axis=0)

            ## if the score is less than "score_pen_tresh" , then the subsample features gets a penalty instead of a reward
            if score_pen_tresh != None:
                if sco<score_pen_tresh:
                    ## the penalty is measured with the distance to the threshold 
                    sco = abs(score_pen_tresh-sco)
                    
            ## update concerned features scores
            for j in range(len(sub_feat)):
                scores[sub_feat[j]] += sco*feature_importances[j] ## the reward/penalty is proportional to feature importnace
            ## print progress
            self.printProgressBar(i + 1, self.nbIter, prefix = 'job running: feature importance - feature selection process ')

        scores = pd.DataFrame(list(scores.values()),columns=['score'],index=list(scores.keys()))
        scores['score'] /= scores['score'].sum()
        scores.sort_values(by='score',ascending=False,inplace=True)
        scores['cum_score'] = scores['score'].cumsum()
        ## display  scores distribution
        plt.figure(figsize=(10,5))
        plt.bar(range(1,scores.shape[0]+1),scores['cum_score'])
        plt.show()
        ## return scores
        return scores



        