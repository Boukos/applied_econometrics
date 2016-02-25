#############################
# IV homework, Econometrics #
# Rudy Gilman               #
# Nov 20, 2015              #
#############################

import numpy as np
import pandas as pd
import plotly
from pandas import DataFrame
import statsmodels.formula.api as smf

### Creating variables ###

#np.random.seed(1234)

# Note: I'm making x1 explicitly correlated with x2, though the R file provided only has x1 explicitly correlated with x3 and x4 (as the R file has it, x1 and x2 are slightly correlated because they both have variance, but we don't know in which direction they're correlated, making the questions below hard to answer). All other variables are set up identically.

x1 = np.random.normal(3.0, .5, 100)
x2 = np.random.normal(2.0, 2.0, 100)
x3 = np.random.normal(2.0, 1.0, 100)
x4 = np.random.normal(0.0, 3.0, 100)
u = np.random.normal(0.0, 1.0, 100)

x1 = x1 + x2 + x3 + x4

y = 5 + 2*x1 - 4*x2 + u

df = DataFrame({'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4, 'u':u, 'y':y})



### 1 ###

# As I've set up the variables, running a regression of y on x1 will give me a negatively biased estimator because x1 and x2 are positively correlated while x2 and y are inversely related. The coefficient will be too low compared to the true value.



### 2 ###

lm = smf.ols(formula="y~x1", data = df).fit()

print(lm.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.306
Model:                            OLS   Adj. R-squared:                  0.299
Method:                 Least Squares   F-statistic:                     43.22
Date:                Fri, 20 Nov 2015   Prob (F-statistic):           2.39e-09
Time:                        22:39:39   Log-Likelihood:                -330.04
No. Observations:                 100   AIC:                             664.1
Df Residuals:                      98   BIC:                             669.3
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      3.3630      1.299      2.588      0.011         0.784     5.942
x1             1.1177      0.170      6.574      0.000         0.780     1.455
==============================================================================
Omnibus:                        4.553   Durbin-Watson:                   1.840
Prob(Omnibus):                  0.103   Jarque-Bera (JB):                3.886
Skew:                          -0.407   Prob(JB):                        0.143
Kurtosis:                       3.519   Cond. No.                         15.2
==============================================================================
"""
# As predicted, this model doesn't give a good result. The true value isn't contained in the 95% confidence interval reported. It's negatively biased.



### 3 ###

# x4 will work better because it has more variation and thus displays a higher correlation with x1 than does x3. 



### 4 ###

lm1_x3 = smf.ols(formula="x1~x3", data = df).fit()
lm1_x4 = smf.ols(formula="x1~x4", data = df).fit()

pred1_x3 = lm1_x3.predict()
pred1_x4 = lm1_x4.predict()

df['pred1_x3'] = pred1_x3
df['pred1_x4'] = pred1_x4

lm2_x3 = smf.ols(formula="y~pred1_x3", data = df).fit()
lm2_x4 = smf.ols(formula="y~pred1_x4", data = df).fit()

print(lm2_x3.summary())
print(lm2_x4.summary())

# Our answer to question 3 is validated. Using x4 as an instrument provides reliably better results than does using x3. I tried a few iterations and all provided a 95% confidence interval containing the true value. Most of the iterations using x3 did, as well, but the intervals were much wider as would be expected. 


