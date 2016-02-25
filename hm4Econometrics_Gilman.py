#####################
# Rudy Gilman       #
# Econometrics hw4  #
# Oct 17, 2015      #
#####################

# This program is in answer to question 14. I first create two variables (x1, x2) which are partially correlated with one another, then I create a third variable from them with some noise added in (y). The standard errors on the beta hats are estimated using the equation given in lecture 6. They are checked against the standard errors given by the statsmodels package in Python.


import numpy as np
import pandas
import plotly
from pandas import DataFrame
import statsmodels.formula.api as smf

data = np.random.multivariate_normal([2,3], [[1.0, .3],[.3, 1.0]], 100)

print(data)

"""
df = DataFrame(data, columns=['x1', 'x2'])

u = np.random.normal(0.0, 2.0)

df['u'] = np.zeros(len(df))

for i in range(len(df.u)):
    df.u[i] = np.random.normal(0.0, 2.0)

df['y'] = 5 + 2*df.x1 + 5*df.x2 + df.u

lm = smf.ols(formula='y ~ x1 + x2', data = df).fit()

fit = lm.predict()

res = df.y - fit

# b hat 1
uvar = sum((res)**2) / 97
sstx1 = sum((df.x1 - np.mean(df.x1))**2)
r = smf.ols(formula='x1 ~ x2', data = df).fit().rsquared
bh1std = np.sqrt(uvar / (sstx1 * (1.0-r)))

# b hat 2
uvar = sum((res)**2) / 97
sstx2 = sum((df.x2 - np.mean(df.x2))**2)
r = smf.ols(formula='x2 ~ x1', data = df).fit().rsquared
bh2std = np.sqrt(uvar / (sstx2 * (1.0-r)))


lm = smf.ols(formula='y ~ x1 + x2', data = df).fit()

"""


