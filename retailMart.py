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
df = pd.read_excel(path+"RetailMart.xlsx")

y = df.PREGNANT
kill=['Unnamed 17','PREGNANT']
X = df[df.columns[df.columns.isin(kill)==False]]

dt = DecisionTreeClassifier(min_samples_split=40, max_depth=1)
dt.fit(X,y)

export_graphviz(dt, out_file='tree.dot')






















