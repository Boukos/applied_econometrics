#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################
# Applied Econometrics, PS4             #
# Rudy Gilman                           #
# March 30, 2016                        #
#########################################

import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os, copy
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import seaborn as sns

path = "/home/rudebeans/Desktop/school_spring2016/applied_econometrics/"

#df = pd.read_stata(path+"poll7080.dta", convert_categoricals=False) dta not loading

df = pd.read_csv(path+"poll7080.csv")

df['intercept'] = 1 # Adding constant


"""
Problem Set 4

This exercise examines the following research question: What is the relationship between changes in air pollution and housing prices? For background on the topic and the data source refer to the paper, “Does Air Quality Matter? Evidence from the Housing Market,” by Kenneth Chay and Michael Greenstone, Journal of Political Economy, April 2005, 376-424.  Please include a concise summary of your empirical results when appropriate.  We will analyze the following data set:

Data Source: poll7080.dta
This STATA data extract is from a combination of the 1972 and 1983 City and County Data Books, the EPA’s Air Quality Subsystem data file, and the Code of Federal Regulations. The data is measured at the county-level in the United States.

Data Notes:
1.  There are 1,000 observations at the U.S. county level.  These are the counties with particulates pollution monitors both at the beginning and end of the 1970s and contain the vast majority of the U.S. population (over 80%).

2. The key variables are:
dlhouse = change in log-housing values from 1970 to 1980 (1980 log-price minus 1970 log-price). 
dgtsp = change in the annual geometric mean of total suspended particulates pollution (TSPs) from 1969-72 to 1977-80 (1977-80 TSPs minus 1969-72 TSPs).
tsp75 = indicator equal to one if the county was regulated by the Environmental Protection Agency (EPA) in 1975 and equal to zero, otherwise.
tsp7576 = indicator equal to one if the county was regulated by the Environmental Protection Agency (EPA) in either 1975 or 1976 and equal to zero, otherwise.
mtspgm74 = annual geometric mean of TSPs in 1974.
mtspgm75 = annual geometric mean of TSPs in 1975.

3. The other relevant variables are:
ddens = 1970-80 change in population density, 
dmnfcg = change in % manufacturing employment, 
dwhite = change in fraction of population that is white, 
dfeml = change in fraction female, 
dage65 = change in fraction over 65 years old, 
dhs = change in fraction with at least a high school degree, 
dcoll = change in fraction with at least a college degree, 
durban = change in fraction living in urban area, 
dunemp = change in unemployment rate, 
dincome = change in income per-capita, 
dpoverty = change in poverty rate, 
vacant70 and vacant80 = housing vacancy rate in 1970 and 1980, 
vacrnt70 = rental vacancy rate in 1970, 
downer = change in fraction of houses that are owner-occupied, 
dplumb = change in fraction of houses with plumbing, 
drevenue = change in government revenue per-capita, 
dtaxprop = change in property taxes per-capita, 
depend = change in general expenditures per-capita, 
deduc = change in fraction of spending on education, 
dhghwy = change in % spending on highways, 
dwelfr = change in % spending on public welfare, 
dhlth = change in % spending on health, 
blt1080 = % of houses built in the last 10 years as of 1980, 
blt2080 = % of houses built in the last 20 years as of 1980, 
bltold80 = % of houses built more than 20 years ago as of 1980.

The remaining variables in the data set are polynomials and interactions of the control variables.

Research Question: Does Air Quality Get Capitalized into Housing Prices?
The outcome of interest is the change in county housing prices during the 1970s. We want to estimate the “causal” effect of air pollution changes on housing price changes. According to hedonic price theory, the housing market may be used to estimate the implicit prices of clean air and the economic value of pollution reductions to individuals (if you’re interested in hedonic pricing, see this article: Rosen, Sherwin, “The Theory of Equalizing Differences,” Chapter 12 in Handbook of Labor Economics, Volume 1, 1986, pp. 641-92. ) .  A statistically significant negative relationship between changes in property values and pollution levels across counties is interpreted as evidence that clean air has economic benefits.

A basic model for the change in housing prices at the county level could be: 
Change in housing price = g(economic shocks, changes in county characteristics, change in air pollution).
"""

print "a. Estimate the relationship between changes in air pollution and housing prices: 1) not adjusting for any control variables; 2) adjusting for the main effects of the control variables listed on the previous page; and 3) adjusting for the main effects, polynomials and interactions of the control variables included in the data set. What do your estimates imply and do they make sense? Describe the potential omitted variables biases. What is the likely relationship between economic shocks and pollution and housing price changes? Using the observable measures of economic shocks (dincome, dunemp, dmnfcg, ddens, durban, blt1080), provide evidence on this.\n\n"

print "Answer: Model 1) with no controls shows a positive correlation btwn increases in pollution and increases in house prices. Adding controls drives this relationship in the direction we would expect--towards a negative correlation. Our estimates make sense when we consider the effects of economic shocks. As we can see in the scatterplot matrix below, increased unemployment and decreased incomes are associated with lower levels of pollution as well as lower house prices. Without controlling for economic shocks, our estimate was biased downwards (ie being confounded by omitted variables working in the opposite direction). It probably still is, as we've only used observable manifestations of economic shock. Model 3) with all the controls is difficult to interpret.\n\n"

### Creating Xs for linear models

# X_1

X_1 = df[['dgtsp', 'intercept']]

# X_2

# bag of words from which to draw control variables of interest
p = "intercept dgtsp tsp75 = indicator equal to one if the county was regulated by the Environmental Protection Agency (EPA) in 1975 and equal to zero, otherwise. tsp7576 = indicator equal to one if the county was regulated by the Environmental Protection Agency (EPA) in either 1975 or 1976 and equal to zero, otherwise. mtspgm74 = annual geometric mean of TSPs in 1974. mtspgm75 = annual geometric mean of TSPs in 1975. 3. The other relevant variables are: ddens = 1970-80 change in population density,  dmnfcg = change in % manufacturing employment,  dwhite = change in fraction of population that is white,  dfeml = change in fraction female,  dage65 = change in fraction over 65 years old,  dhs = change in fraction with at least a high school degree,  dcoll = change in fraction with at least a college degree,  durban = change in fraction living in urban area,  dunemp = change in unemployment rate,  dincome = change in income per-capita,  dpoverty = change in poverty rate,  vacant70 and vacant80 = housing vacancy rate in 1970 and 1980,  vacrnt70 = rental vacancy rate in 1970,  downer = change in fraction of houses that are owner-occupied,  dplumb = change in fraction of houses with plumbing,  drevenue = change in government revenue per-capita,  dtaxprop = change in property taxes per-capita,  depend = change in general expenditures per-capita,  deduc = change in fraction of spending on education,  dhghwy = change in % spending on highways,  dwelfr = change in % spending on public welfare,  dhlth = change in % spending on health,  blt1080 = % of houses built in the last 10 years as of 1980,  blt2080 = % of houses built in the last 20 years as of 1980,  bltold80 = % of houses built more than 20 years ago as of 1980"

p = p.split() # list of words
X_2_cols = (df.columns[df.columns.isin(p)==True]) # basic control variables
X_2 = df[X_2_cols]

# X_3

X_3 = df[df.columns[df.columns != 'dlhouse']]

# Returns df with variables and importance, descending
def get_imp(X,y):
    #rf = RandomForestClassifier()
    rf = DecisionTreeRegressor(random_state=9)
    rf.fit(X, y)
    imp_var = rf.feature_importances_
    imp_var = pd.DataFrame({'variable':X.columns, 'imp':imp_var}).sort('imp', ascending=False)
    return(imp_var)
    
var = get_imp(X_3.fillna(0), df.dlhouse)
imp_var = var[var.imp > 0.0005] # only keeping variables w some explanatory power

kill = ['lhouse80', 'house80'] # combined w numbers for 70, these perfectly predict dlhouse
X_3 = X_3[imp_var.variable]
X_3 = X_3[X_3.columns[X_3.columns.isin(kill)==False]]
X_3['intercept'] = 1 # adding back in intercept and dgtsp, as both were removed in Imp filter
X_3['dgtsp'] = df.dgtsp

print "X_1"
print "\n"
lm = sm.OLS(df.dlhouse, X_1, missing='drop').fit()
print lm.summary()
print "1) dgtsp coefficient: "+str(lm.params['dgtsp'])
print "\n\n\n"

print "X_2"
print "\n"
lm = sm.OLS(df.dlhouse, X_2, missing='drop').fit()
print lm.summary()
print "2) dgtsp coefficient: "+str(lm.params['dgtsp'])
print "\n\n\n"

print "X_3"
print "\n"
lm = sm.OLS(df.dlhouse, X_3, missing='drop').fit()
print lm.summary()
print "3) dgtsp coefficient: "+str(lm.params['dgtsp'])
print "\n\n\n"

cols = ['dincome','dunemp', 'dgtsp','dmnfcg', 'dlhouse']

pp = df[cols].dropna()
sns.set(style="ticks", color_codes=True)
#iris = sns.load_dataset("iris")

g = sns.PairGrid(pp)
g = g.map_upper(plt.scatter, alpha=.5)
g = g.map_lower(sns.kdeplot, cmap="Blues_d")
g = g.map_diag(sns.distplot)
#g = sns.pairplot(pp, kind='reg')

# Rescaling axis. Why doesn't Seaborn do this for me automatically?
axes = g.axes
for i in range(len(cols)):
    axes[i,i].set_ylim(np.min(pp[cols[i]]),np.max(pp[cols[i]]))
    axes[i,i].set_xlim(np.min(pp[cols[i]]),np.max(pp[cols[i]]))
#g.set(ylim=(0, 9))
#g.set(xlim=(0, 9)) sets all axes to this extent

plt.show()

print "b. Suppose that federal EPA pollution regulation is a potential instrumental variable for pollution changes during the 1970s. What are the assumptions required for 1975-1976 regulatory status, tsp7576, to be a valid instrument for pollution changes when the outcome of interest is housing price changes? Provide evidence on the relationship between the regulatory status indicator and the observable economic shock measures. Interpret your findings.\n\n"

print "Answer: tsp7576 must be strong, ie correlated w dgtsp, and valid, ie not correlated w dlhouse except through dgtsp. As we can see in the correlation matrix below, tsp7576 is only slightly correlated w our economic shock variables, making it potentially a good IV.\n\n"

shock_cols = ['tsp7576','dincome','dunemp', 'dmnfcg', 'ddens', 'durban', 'blt1080']

d = df[shock_cols]
cor = d.corr()
print cor

#lm = sm.OLS(df.tsp7576, df[shock_cols], missing='drop').fit()
#print lm.summary()
#print "\n\n"

print "\n\n c. Document the “first-stage” relationship between regulation (tsp7576) and air pollution changes and the “reduced-form” relationship between regulation and housing price changes, using the same three specifications you used in part a). Interpret your findings. How does two-stage least squares use these two equations?  Now estimate the effect of air quality changes on housing price changes using two-stage least squares and the tsp7576 indicator as an instrument for the three specifications. Interpret the results. Now do the same using the 1975 regulation indicator, tsp75, as an instrumental variable. Compare the findings.\n\n"

print "Answer: the negative effect of regulation on air pollution decreases as we add controls (relationship gets less negative). The positive effect of regulation on changes in house prices decreases as we add controls. In terms of IV, our 7576 and 75 results are similar, both around -.003 (elasticity of -0.3) for uncontrolled and basically controlled regressions. As we showed before, the IV is only slightly correlated with economic shock variables, so the model isn't particularly sensitive to control specification. Estimates for model w all interactions + higher-order terms remains difficult to interpret. The reduced form results show that house prices in treated counties are 2-4% higher than they would be otherwise. First-stage results suggest that treatment results in pollution decreases of 5-10%.\n\n"

# FOr 7576
X_1 = df[['intercept', 'tsp7576']]
kill = ['dgtsp', 'tsp75']
X_2 = X_2[X_2.columns[X_2.columns.isin(kill)==False]]
X_3 = X_3[X_3.columns[X_3.columns.isin(kill)==False]]
X_3['tsp7576'] = df.tsp7576

Xs=[X_1, X_2, X_3]
Xs_str=['X_1', 'X_2', 'X_3']

for i in range(len(Xs)):
    print Xs_str[i]
    X = Xs[i]
    first = sm.OLS(df.dgtsp, X, missing="drop").fit()
    print first.summary()
    print '\n'
    print Xs_str[i]+": first stage tsp7576 coefficient:"
    print first.params['tsp7576']
    reduced = sm.OLS(df.dlhouse, X, missing="drop").fit()
    print reduced.summary()
    print '\n'
    print Xs_str[i]+": reduced form tsp7576 coefficient:"
    print reduced.params['tsp7576']  
    # IV
    dgtsp_hat = first.predict(X)
    X['dgtsp_hat'] = dgtsp_hat
    X = X.drop(['tsp7576'], axis=1)
    IV = sm.OLS(df.dlhouse, X, missing="drop").fit()
    print Xs_str[i]+ ": IV estimate: "+ str(IV.params['dgtsp_hat'])
    print IV.summary()    
    print '\n'
    
 
 
# for 75  
X_1 = df[['intercept', 'tsp75']]
kill = ['dgtsp', 'tsp7576']
X_2 = X_2[X_2.columns[X_2.columns.isin(kill)==False]]
X_2['tsp75'] = df.tsp75
X_3 = X_3[X_3.columns[X_3.columns.isin(kill)==False]]
X_3['tsp75'] = df.tsp75

Xs=[X_1, X_2, X_3]
Xs_str=['X_1', 'X_2', 'X_3']

for i in range(len(Xs)):
    print Xs_str[i]
    X = Xs[i]
    first = sm.OLS(df.dgtsp, X, missing="drop").fit()
    print first.summary()
    print '\n'
    print Xs_str[i]+": first stage tsp75 coefficient:"
    print first.params['tsp75']
    reduced = sm.OLS(df.dlhouse, X, missing="drop").fit()
    print reduced.summary()
    print '\n'
    print Xs_str[i]+": reduced form tsp75 coefficient:"
    print reduced.params['tsp75']  
    # IV
    dgtsp_hat = first.predict(X)
    X['dgtsp_hat'] = dgtsp_hat
    X = X.drop(['tsp75'], axis=1)
    IV = sm.OLS(df.dlhouse, X, missing="drop").fit()
    print Xs_str[i]+ ": IV estimate: "+ str(IV.params['dgtsp_hat'])
    print IV.summary()    
    print '\n' 


print("d. In principle, the 1975 regulation indicator variable, tsp75, should be a discrete function of pollution levels in 1974. Specifically, the EPA is supposed to regulate those counties in 1975 who had either an annual geometric mean of TSPs above 75 units (μg/m3) or a 2nd highest daily concentration above 260 units in 1974. Describe how one could use this discontinuity in treatment assignment to derive alternative estimates of the capitalization of pollution changes. Under what conditions will these estimates be valid? Describe the graphical analysis you would use to examine the validity of these conditions.\n\n")


print "Answer: We could take only the section of the dataframe + and - a certain delta around the threshhold value of pollution in 1974. Then we could construct a Wald Estimator with it, i.e. find the difference in price changes and pollution changes for a group just under the threshhold, as well as for a group just over the threshhold, then divide the differences to find change in dlhouse per change in TSPs. I'd want to see that the discontinuity wasn't fuzzy, so a graph like that below would be helpful. I'd want to verify that group means were similar in below-threshhold and above-threshhold groups. \n\n"


y = 'tsp75'
x = 'mtspgm74'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(df[x], df[y], c='blue', alpha=0.25)
ax.set_title(str(x)+' '+str(y))  
ax.set_ylabel(str(y))
ax.set_xlabel(str(x))
#ax.set_ylim([0,7])
#ax.set_xlim([500,1500])
#ax.legend([y, y2])
plt.show()
fig.clf() 
print '\n'


print "e. Describe (in words) the theoretical reasons why the effects of pollution changes on housing price changes may be heterogeneous.  Under what assumptions will two-stage least squares identify the average treatment effect (ATE)? What is the economic interpretation of ATE in the context of hedonic theory? If ATE is not identified, describe what may be identifiable with two-stage least squares estimation. Under what conditions is this effect identified? Give some intuition on what this effect may represent when one uses EPA regulation as an instrument.\n\n"

print "Answer: Small, unnoticable changes in pollution might have no effect on changes in price, while larger changes in pollution may have large effects on price changes, perhaps increasing at an increasing rate as pollution changes become noticable, then at a diminishing rate when jumps in pollution are extreme. 2SLS will identify the ATE if effects of pollution changes on price changes are homogenous. The ATE in the context of the hedonic theory represents the amount people have revealed themselves willing to pay for clean air. \nIn this case, however, we've only IDed the LATE, as our results are only applicable around the pollution threshhold. The LATE is the effect of pollution changes on house price changes in the small window around the regulatory threshhold for those cities which were induced into cleaning up as a result of regulations, but wouldn't have done so otherwise. We should also be aware of selection effects: People in the highly-polluted areas eligable for treatment may be less sensitive to pollution, so we're measuring the effect amongst a demographic that may not be representative of the greater population.\n\n"



print "f. Now provide a concise synthesis/summary of your results.  Discuss the “credibility” of the research designs underlying the results.\n\n"


print "Answer: In our simple OLS models, we measured a slightly positive effect of changes in pollution on changes in house price. On the face of it, this runs counter to what we expect the relationship to be. When we consider the omitted variable of economic shocks, however, we see how this paradoxical result came about--negative economic shocks decrease both pollution and house prices. Controlling for a few manifestations of economic shocks (unemployment, changes in income, etc) helped ameliorate OVB to some extent. \nTo more thoroughly purge our model of OVB, we used an IV approach, (hopefully) capturing only the variation in dgtsp uncorrelated with our confounding omitted variable. Using this approach, we captured a more expected result: A 1% increase in pollution is associated with an approximate .3% decrease in price. This result was robust to our selection of control variables, probably bc our IV was uncorrelated with the OV. This strikes me as a credible approach. \n\n"


