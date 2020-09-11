# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:47:55 2020

@author: hikne
"""

###################################
###                             ### 
####         Librairies        ####
###                             ###
###################################
#import dataiku
import pandas as pd, numpy as np
#from dataiku import pandasutils as pdu
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
from scipy.stats import skew
from itertools import product
from scipy.stats import ks_2samp
from sklearn import preprocessing


#------------------------------------------#
#                                          # 
##   Feature cross-correlation            ##
#                                          #
#------------------------------------------#

def plot_density(s,name):
    print('skewness= {}'.format(abs(skew(s))))
    plt.figure(figsize=(7,3))
    sns.distplot(s, hist=True, kde=True, 
                 bins=np.linspace(s.min(),s.max(),100), color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})
    plt.title(name)
    plt.show()
    
    
#------------------------------------------#
#                                          # 
##   Feature cross-correlation            ##
#                                          #
#------------------------------------------#

def Display_cross_correlation(df,features):
    """
    @ Description: Display feature correlation matrix.
    @params:
        data              - Required  : dataset (DataFrame)
        features          - Required  : features list (List)
    """
    plt.style.use('default')
    plt.figure(figsize=(24,20))
    ax=sns.heatmap(df[features].corr(),xticklabels=features,yticklabels=features, cmap="coolwarm", square=True,annot=True, fmt='.2f', 
                linewidths=.1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")
    plt.show()


    
#------------------------------------------#
#                                          # 
##   Feature=func(target) distribution    ##
#                                          #
#------------------------------------------#

def displayDist(df,features,target,kind='box'):
    """
    @ Description: Plot feature distributon by target values.
    @params:
        data              - Required  : dataset (DataFrame)
        features          - Required  : features list (List)
        target            - Required  : target name (String)
        kind              - Optional  : Chart kind (String) <possible values=['box','kde']>
    """
    plt.style.use('default')
    if len(features)>9:
        features=np.random.choice(a=features,replace=False,size=9)
    plt.figure(figsize=(15,15))
    if kind=='box':
        for i,col in enumerate(features,start=1):
            plt.subplot(3,3,i)
            sns.boxplot(x=target,y=col,data=df)
    else:
        features=np.random.choice(a=features,replace=False,size=4)
        sns.pairplot(df[list(features)+[target]], hue=target)
    plt.show()

    
#------------------------------------------#
#                                          # 
##      Feature-target correlation        ##
#                                          #
#------------------------------------------#    
    
def feature_target_correlation(data,features,target='label',tresh=.1):
    """
    @ Description: Display feature to target correlation, and return features with correlation above a given treshold.
    @params:
        data              - Required  : dataset (DataFrame)
        num_cols          - Required  : feature names (List/Array).
        target            - Required  : target name in the dataset (String) <default='label'>.
        tresh             - Optional  : correlation treshold for feature to be returned (Float ) <default=0.1>.
        
    @ Output:
        corr_features:  most correlated features (above the treshold) (Array)
    
    """
    corr={'col':features,'corr':[]}
    
    for col  in features:
        corr['corr'].append(np.corrcoef(data[col], data[target])[0,1])
    corr=pd.DataFrame(corr)
    ##
    #corr['range']=pd.cut(corr['corr'],bins=np.linspace(corr['corr'].min()-.01,corr['corr'].max(),11)).astype(str)
    ## charts
    plt.style.use('default')
    #s=corr['range'].value_counts(1)
    #plt.pie(s,labels=s.index,autopct=lambda p: '{:.2f}%'.format(p))
    #plt.show()
    ##
    corr.plot.hist(by='corr',bins=150,edgecolor='y',figsize=(8,4))
    corr['keep']=corr['corr'].apply(lambda x:abs(x)>tresh)
    return corr.loc[corr['keep'],'col'].values




    
#------------------------------------------#
#                                          # 
##       Feature-target pie chart         ##
#                                          #
#------------------------------------------#        
    
def plotPie(df,col,target='SALES_ORGANIZATION_CODE'):
    print('corr {}* Y {}'.format(col,round(np.corrcoef(df[col],df["Y"])[0,1],4)))
    plt.style.use('default')
    plt.figure(figsize=(9,5))
    for i,vk in enumerate(df[target].unique(),start=1):
        plt.subplot(2,3,i)
        s=df.loc[df[target]==vk,col].value_counts(1)
        plt.pie(s,labels=s.index,autopct=lambda p: '{:.2f}%'.format(p))
        plt.title(vk)
    plt.tight_layout()
    plt.show()
    
    
#------------------------------------------#
#                                          # 
##       Feature-target kde chart         ##
#                                          #
#------------------------------------------#     
  
    
    
def FeatSubDistance(df,col,t=None,e=1,scale=False):
    # transformations
    def slog(x,e=1):
        if abs(x+e)<1:
            return 0
        return np.sign(x)*np.log(abs(x+e))

    transf={'slog':slog,
        'log':lambda x,e:np.log(x+e),
        'log10':lambda x,e:np.log10(x+e),
        'root':lambda x,e:pow(x,1/e),
        'inv':lambda x,e,:1/(abs(x)**e+1)}
    # sample size
    size=df['SALES_ORGANIZATION_CODE'].value_counts().min()
    dfu=pd.DataFrame([])
    #------------
    # transform & scale data
    #-------------------
    for vk in df['SALES_ORGANIZATION_CODE'].unique():
        dfx=df.loc[df['SALES_ORGANIZATION_CODE']==vk,['SALES_ORGANIZATION_CODE',col]].copy().sample(n=size)
        # if requested -> make transfomration 
        if t!=None:
            dfx[col]=dfx[col].apply(transf[t],e=e).to_frame(col)
        # if requested -< scale data 
        if scale:
            dfx[col]=preprocessing.MinMaxScaler().fit_transform(dfx[[col]])
        dfu=pd.concat([dfu,dfx],axis=0)
    #-------------
    # compute 
    #-------------
    vks=sorted(dfu['SALES_ORGANIZATION_CODE'].unique().tolist())
    d=np.eye(len(vks))
    for vk1,vk2 in product(vks,vks):
        d[vks.index(vk1),vks.index(vk2)]=ks_2samp(dfu.loc[dfu['SALES_ORGANIZATION_CODE']==vk1,col],dfu.loc[dfu['SALES_ORGANIZATION_CODE']==vk2,col])[0]
    #-----------
    # plot distance matrix
    #-----------
    plt.figure(figsize=(7,5)) 
    h=plt.pcolor(d,)
    plt.xticks(.5+np.arange(len(vks)),vks)
    plt.yticks(.5+np.arange(len(vks)),vks)  
    ## add text to heatmap
    for y in range(d.shape[0]):
        for x in range(d.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % d[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    plt.colorbar(h)
    plt.title(f'{col} distribution distance [t={t},e={e}]')
    plt.tight_layout()
    plt.show()