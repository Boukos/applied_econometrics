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

path = "/home/rudebeans/Desktop/school_spring2016/applied_econometrics/"

df = pd.read_stata(path+"smoking2.dta", convert_categoricals=False).sample(500)
cols = ['dbirwt','dfeduc','dmage']
cm = np.corrcoef(df[cols])


sns.pairplot(df[cols])
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()



"""
df_lg = pd.read_pickle('df_full_lg.txt')
kill = ['tobacco', 'dbirwt', 'death'] 
X = df_lg[df_lg.columns[df_lg.columns.isin(kill) == False]]
y = df_lg.tobacco
scoring = 'accuracy'

dt = DecisionTreeClassifier()
scores = cross_val_score(estimator=dt, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("DT")
print(np.mean(scores))
print("\n\n\n")

export_graphviz(dt, out_file='treeTest.dot')


dot_data = StringIO() 
export_graphviz(dt, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("testTree.pdf") 




# rf 
rf = RandomForestClassifier(criterion='entropy', min_samples_split=40, random_state=99)
#rf = RandomForestClassifier()
#rf.fit(X, y)
#scores = cross_val_score(estimator=rf, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
#print("RF")
#print(np.mean(scores))
#print("\n\n\n")

# logit
#lg = LogisticRegression(C=1000.0, random_state=0)
lg = LogisticRegression(random_state=3)
#lg.fit(X,y)
#scores = cross_val_score(estimator=lg, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
#print("LOGIT")
#print(np.mean(scores))
#print("\n\n\n")


# ensemble
estimators = [('lg',lg),('rf',rf)]#,('knn',knn),('svM',svM)
eclf = VotingClassifier(estimators=estimators, voting='hard')
#scores = cross_val_score(estimator=eclf, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
#print("ENSEMBLE")
#print(np.mean(scores))
#print("\n\n\n")

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

#knn
#knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn = KNeighborsClassifier()
#knn.fit(X,y)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=5, n_jobs=-1, scoring=scoring)
print("KNN")
print(np.mean(scores))
print("\n\n\n")
"""

