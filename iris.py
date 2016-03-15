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
from sklearn import datasets
from sklearn.decomposition import PCA
import subprocess
from sklearn.metrics import precision_recall_curve


iris = datasets.load_iris()
columns = ['sepal_length','sepal_width','petal_length','petal_width']
X = pd.DataFrame(iris.data[:, :4], columns=columns)

class_names=['versicolour','virginica']
# Iris-Setosa, Iris-Versicolour, Iris-Virginica
y = pd.DataFrame(iris.target, columns=['type'])

# only using setosa and versicolour
df = pd.concat([X,y], axis=1)
df = df[df.type.isin([1,2])]


X = df[df.columns[df.columns!='type']]
y = df.type

def make_0(x):
    if x == 2:
        return 0
    else: return 1
y = y.map(make_0)

dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, max_depth=2)

dt.fit(X,y)

pred = dt.predict_proba(X)
pred = [e[1] for e in pred]

prc = precision_recall_curve(y, pred)


print("sentence {} and {}".format(1, 3))


"""
print(df)

def vis_tree(tree, feature_names, file_name):
    with open(file_name+'.dot', 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True)
    command = ["dot", "-Tpng", file_name+'.dot', "-o", file_name+".png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

vis_tree(dt, columns, "dt")

cols = [['sepal_length','sepal_width'],['sepal_width','petal_length'],['petal_length','petal_width']]
for i in range(3):
    x = X[cols[i]]
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=2)
    dt.fit(x,y)
    vis_tree(dt, cols[i], "dt"+str(i))
"""
    

    


