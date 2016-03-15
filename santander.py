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

#df = pd.read_pickle('df_with_p.txt')

path = "/home/rudebeans/Downloads/"

df = pd.read_csv(path+"train.csv")
re = df[df.TARGET==1]
re_non = df[df.TARGET==0].sample(n=10000, random_state=1985)

re_df = pd.concat([re,re_non], axis=0)

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
X = re_df[re_df.columns[re_df.columns!='TARGET']]#.iloc[:,5:20]
#X = re_df[['var38','var15']]

# Returns df with variables and importance, descending
def get_imp(X,y):
    rf = RandomForestClassifier(criterion='entropy', min_samples_split=40, n_estimators=50, random_state=99, max_depth=10)
    rf.fit(X, y)
    imp_var = rf.feature_importances_
    imp_var = pd.DataFrame({'variable':X.columns, 'imp':imp_var}).sort('imp', ascending=False)
    return(imp_var)

scoring = 'accuracy'

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

# c = 0.1
important = get_imp(X,y)
cols=sorted(list(important.imp[:30]))
#Xs = X[important.variable[:16]]

lg = LogisticRegression(C=0.1, random_state=0)
"""
C_range = [0.0001,0.01,0.1, 1.0, 10.0, 1000.0]
param_grid = [{'C': C_range}]
gs = GridSearchCV(estimator=lg, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X,y)
print("GS")
print(gs.best_score_)
print(gs.best_params_)
print('\n\n')
"""
lg.fit(X, y)
scores = cross_val_score(estimator=lg, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("LG")
print(np.mean(scores))
print("\n\n\n")

# ensemble
estimators = [('lg',lg),('rf',rf)]#,('knn',knn),('svM',svM)
eclf = VotingClassifier(estimators=estimators, voting='hard')
scores = cross_val_score(estimator=eclf, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("ENSEMBLE")
print(np.mean(scores))
print("\n\n\n")


"""
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
