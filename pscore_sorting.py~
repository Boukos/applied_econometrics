#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################
# Applied Econometrics, PS1             #
# Rudy Gilman                           #
# Feb 20, 2016                          #
#########################################

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

path = "/home/rudebeans/Desktop/school_spring2016/applied_econometrics/"

#os.path.dirname(__file__)

df = pd.read_stata(path+"smoking2.dta", convert_categoricals=False)
df = df.sample(n=20000, random_state=1985)

print(df)

X = pd.read_pickle('bigX.txt')

# Running logit w all variables
kill = ['tobacco']      
x = X.loc[:,X.columns[X.columns.isin(kill) == False]]
lg = sm.Logit(X.tobacco, x).fit(method='bfgs')
print(lg.summary())

# Running logit again w only significant variables
sig = lg.tvalues[(np.absolute(lg.tvalues)>2)].index
x = x.loc[:,x.columns[x.columns.isin(sig)]]
lg = sm.Logit(X.tobacco, x).fit()
print(lg.summary())

# Predicted logit values (propensity score) back to df
pred = lg.predict()
df['p'] = pred

df_smoke = df[df.tobacco == 1]
df_noSmoke = df[df.tobacco == 0]

print(df_smoke.p.describe())
print(df_noSmoke.p.describe())




def optimize_propensity_groups(df, threshhold):

    def assign_groups(df):
        num=2 
        ls = sorted(list(df.p))
        l = len(ls)/num

        def get_group(x):
            for i in range(num):
                lower = ls[l*i]
                if x >= lower and x <= ls[l*i+l-1]:
                    return lower             
        p = df.p           
        pg = p.map(get_group)
        df['pg'] = pg
        return df

    df = assign_groups(df) # First assignment of groups

    def get_scores(df):
        tots=[]
        gs = sorted(df.pg.unique())
        for i in range(len(df.pg.unique())):
            groupId = gs[i]
            d = df[df.pg==groupId]
            d_smoke = d[d.tobacco==1]
            d_noSmoke = d[d.tobacco==0]     
            gScore = 0 
            for c in range(len(d.columns)):
                var = d.columns[c]
                if var not in ['pg','tobacco']:      
                    try:
                        t = ttest_ind(d_smoke[var], d_noSmoke[var])[0]
                    except:
                        t = 2      
                    if np.abs(t) < 1.96:
                        gScore += 1            
            groupScore = float(gScore) / float((len(d.columns)-2))
            tots.append((groupId, groupScore))
        return(tots)

    def update_groups(df): # Performs one update pass across groups
        scores = get_scores(df)
        f = pd.DataFrame({})
        for i in range(len(scores)):
            groupId = scores[i][0]
            s = df[df.pg==groupId]
            if scores[i][1] < threshhold:
                s = df[df.pg==groupId]
                s = assign_groups(s)
                f=pd.concat([f, s], axis=0)
            else:
                f=pd.concat([f, s], axis=0)
        return f
            
    def calc_final_score(df):
        scores = get_scores(df)
        groupNames = [i[0] for i in scores]
        groupScores = [i[1] for i in scores]
        match = pd.Series(groupScores)
        match = match[match>=threshhold]
        finalScore = float(len(match)) / float(len(groupScores))
        return finalScore
        
    while calc_final_score(df) < threshhold: # Updating groups until threshhold met
        df = update_groups(df)

    print(calc_final_score(df))
    print(len(get_scores(df)))   
    print len(df.pg.unique())

    return df

optimize_propensity_groups(df, 0.7)


