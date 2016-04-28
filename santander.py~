from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os, copy
from causalinference import CausalModel
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#import pydot 
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

#df = pd.read_pickle('df_with_p.txt')

path = "/home/rudebeans/Downloads/"

df = pd.read_csv(path+"train.csv")
re = df[df.TARGET==1]
re_non = df[df.TARGET==0].sample(n=15000, random_state=1985)

re_df = pd.concat([re,re_non], axis=0)

#re_df = df

"""
# Deleting columns w no variation
cols = list(re_df.columns)
for variable in re_df.columns:
    if np.std(re_df[variable]) == 0:
        print str(variable)
        cols.remove(str(variable))   
re_df = re_df[cols]
"""
y = re_df['TARGET']
X = re_df[re_df.columns[re_df.columns!='TARGET']]


# Returns df with variables and importance, descending
def get_imp(X,y):
    rf = RandomForestClassifier(criterion='entropy', min_samples_split=40, n_estimators=50, random_state=99, max_depth=10)
    rf.fit(X, y)
    imp_var = rf.feature_importances_
    imp_var = pd.DataFrame({'variable':X.columns, 'imp':imp_var}).sort('imp', ascending=False)
    return(imp_var)

scoring = 'roc_auc'

"""
#knn
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
#knn = KNeighborsClassifier()
knn.fit(X,y)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("KNN")
print(np.mean(scores))
print("\n\n\n")


# rf 
rf = RandomForestClassifier(criterion='entropy', min_samples_split=40, random_state=99, n_estimators=20, max_depth=5)
#rf = RandomForestClassifier()
rf.fit(X, y)
scores = cross_val_score(estimator=rf, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("RF")
print(np.mean(scores))
print("\n\n\n")
"""
#c = 0.1
important = get_imp(X,y)
cols=sorted(list(important.imp[:30]))

Xs = X[important.variable[:16]] # try 8 or 9

# returns df w only interacted, higher-order variables
def get_interact_higher(df):
    sig = Xs.columns
    # Interacting each significant variable with every other significant variable
    for i in range(len(sig)): 
        for r in range(len(sig)):
            if i < r:
                colName = str(sig[i]) + str(sig[r])
                df[colName] = df[sig[i]] * df[sig[r]]    
                                       
    # Making higher order variables up to ^2
    for i in range(len(sig)): 
        for r in range(2,3):
            df[str(sig[i])+str(r)] = (df[sig[i]])**r
    dup = []
    for i in range(len(df.columns)): 
        # Deleting identical cols
        for r in range(len(df.columns)):
            if (i < r) and (list(df[df.columns[i]]) == list(df[df.columns[r]])):
                dup.append(df.columns[i])  
        # Deleting columns w no variation
        if np.std(df[df.columns[i]]) == 0:
            print df.columns[i]
            dup.append(df.columns[i])   
    return df

Xs = get_interact_higher(Xs)
Xs.to_pickle('Xs.txt')
print('done interacting')

dt = DecisionTreeClassifier(criterion='entropy', max_depth=1)

ada = AdaBoostClassifier(base_estimator=dt, n_estimators=500, learning_rate=0.07, random_state=3) # params already optimized

"""
scores = cross_val_score(estimator=ada, X=Xs, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("ada")
print(np.mean(scores))
print("\n\n\n")
"""
"""
lr = [0.05,.07, .08] 
ne = [200, 300, 500] 

param_grid = [{'learning_rate': lr, 'n_estimators':ne}]
gs = GridSearchCV(estimator=ada, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=-1)
gs.fit(Xs,y)
print("GS")
print(gs.best_score_)
print(gs.best_params_)
print('\n\n')


lg = LogisticRegression(C=0.1, random_state=0)
#lg.fit(Xs, y)
scores = cross_val_score(estimator=lg, X=Xs, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("LG")
print(np.mean(scores))
print("\n\n\n")
"""

# XGB
xgb = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

"""
scores = cross_val_score(estimator=clf, X=Xs, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("XGB")
print(np.mean(scores))
print("\n\n\n")
"""

# ensemble
estimators = [('ada',ada),('xgb',xgb)]#,('knn',knn),('svM',svM)
eclf = VotingClassifier(estimators=estimators, voting='soft')
scores = cross_val_score(estimator=eclf, X=Xs, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("ENSEMBLE")
print(np.mean(scores))
print("\n\n\n")


"""
X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

# predicting
y_pred= clf.predict_proba(X_test)[:,1]




#svm takes awhile
svM = svm.SVC()
#svM.fit(Xs,y)
scores = cross_val_score(estimator=svM, X=Xs, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("SVM")
print(np.mean(scores))
print("\n\n\n")


C_range = [0.0001,0.01,0.1, 1.0, 10.0, 1000.0]
param_grid = [{'C': C_range}]
gs = GridSearchCV(estimator=lg, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X,y)
print("GS")
print(gs.best_score_)
print(gs.best_params_)
print('\n\n')

depth_range = [None,1,2,5,10,15]
forest_n_range = [20,50,100,200]
param_grid = [{'max_depth': depth_range, 'n_estimators': forest_n_range}]
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
#scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)

#print(scores)
gs.fit(X,y)
print("RF")
print(gs.best_score_)
print(gs.best_params_)
print('\n\n')



sc =[]
cutoff = []
for i in cols:
    important = important[important.imp>=i]
    X=X[important.variable]
    print(X.columns)
    scoring = 'accuracy'
    # logit
    #lg = LogisticRegression(C=1000.0, random_state=0)
    lg = LogisticRegression(random_state=3)
    #lg.fit(X,y)
    scores = cross_val_score(estimator=lg, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
    print("LOGIT")
    print(i)
    print(np.mean(scores))   
    sc.append(np.mean(scores))
    cutoff.append(i)
    print("\n\n\n")

results = pd.DataFrame({'score':sc, 'cutoff':cutoff})
print(results)



# ensemble
estimators = [('lg',lg),('rf',rf)]#,('knn',knn),('svM',svM)
eclf = VotingClassifier(estimators=estimators, voting='hard')
scores = cross_val_score(estimator=eclf, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("ENSEMBLE")
print(np.mean(scores))
print("\n\n\n")



dt = DecisionTreeClassifier()
scores = cross_val_score(estimator=dt, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("DT")
print(np.mean(scores))
print("\n\n\n")

C_range = [0.0001,0.01,0.1, 1.0, 10.0, 1000.0]
depth_range = [2,5,10,15]
forest_n_range = [50,100,300]
param_grid = [{'lg__C': C_range, 'rf__max_depth': depth_range, 'rf__n_estimators': forest_n_range}]
gs = GridSearchCV(estimator=eclf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

gs.fit(X,y)
print("ENSEMBLE")
print(gs.best_score_)
print(gs.best_params_)
print('\n\n')

####################################
max_depth_range = [3,15]
n_estimators_range=[50,150]
param_grid = [{'max_depth': max_depth_range, 'n_estimators':n_estimators_range}]
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)

#gs.fit(X,y)
print("random forest")
print(np.mean(scores))
#print(gs.best_score_)
#print(gs.best_params_)
print('\n\n')


#svm takes awhile
svM = svm.SVC()
#svM.fit(X,y)
scores = cross_val_score(estimator=svM, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("SVM")
print(np.mean(scores))
print("\n\n\n")


# grid search
param_range = [0.0001, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=svM, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

print(gs)

gs.fit(X,y)
print(gs.best_score_)
print(gs.best_params_)


# pipeline
# this pipe gives worse score than simple lg
lg_pipe = Pipeline([('scl',StandardScaler()), ('pca', PCA(n_components=2)), ('lg',LogisticRegression(random_state=3))])
lg_pipe.fit(X,y)
scores = cross_val_score(estimator=lg_pipe, X=X, y=y, cv=5, n_jobs=-1)
print(np.mean(scores))


"""

