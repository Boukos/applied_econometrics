######################
# HW 4, Econometrics #
# Rudy Gilman        #
# Nov 15, 2015       #
######################

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.formula.api import logit, probit, glm
#import plotly
#import plotly.plotly as py          You need the Plotly api for these
#import plotly.graph_objs as go      so I'm just commenting them out. 
#                                    It's just for creating the graph


###
# 1
"""
We want something like: y = a1 + b1*x + d2*a2 + d2*b2*x + d3*a3 + d3*b3*x + u
but this will show three possibly disjointed lines. Let's impose restrictions to ensure that the knots join.

First, we know that at x=18, the first two sections must have the same y-value:
 
a1 + b1*18  = a1 + b1*18 + d2*a2 + d2*b2*18
-d2*a2 = d2*b2*18
a2 = -b2*18

Next, we know that at x=22 the last two sections must have the same y-value:

a1 + b1*x + d2*a2 + d2*b2*x = a1 + b1*x + d2*a2 + d2*b2*x + d3*a3 + d3*b3*x
-d3*a3 = d3*b3*x
a3 = -b3*22

Plugging these into the original equation:

y = a1 + b1*x + d2*(-b2*18) + d2*b2*x + d3*(-b3*22) + d3*b3*x + u
y = a1 + b1*x - d2*b2*18 + d2*b2*x - d3*b3*22 + d3*b3*x + u
y = a1 + b1*x + d2*b2*(x - 18) + d3*b3*(x - 22) + u
"""



###
# 2

df = pd.read_csv("/home/rudebeans/Downloads/Solow-1957-technical-change.csv")

df['y'] = df.q / df.A
df['lny'] = np.log(df.y)
df['lnk'] = np.log(df.k)
df['ik'] = 1 / df.k



lm = smf.ols(formula='y~lnk', data=df).fit()
print(lm.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.944
Model:                            OLS   Adj. R-squared:                  0.942
Method:                 Least Squares   F-statistic:                     651.8
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           5.94e-26
Time:                        15:15:04   Log-Likelihood:                 144.10
No. Observations:                  41   AIC:                            -284.2
Df Residuals:                      39   BIC:                            -280.8
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.4524      0.009     50.109      0.000         0.434     0.471
lnk            0.2381      0.009     25.531      0.000         0.219     0.257
==============================================================================
Omnibus:                       17.592   Durbin-Watson:                   0.146
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               21.120
Skew:                           1.691   Prob(JB):                     2.59e-05
Kurtosis:                       3.962   Cond. No.                         15.6
==============================================================================
"""



lm2 = smf.ols(formula='y~ik', data=df).fit()
print(lm2.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.949
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     728.0
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           7.70e-27
Time:                        15:19:30   Log-Likelihood:                 146.25
No. Observations:                  41   AIC:                            -288.5
Df Residuals:                      39   BIC:                            -285.1
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.9197      0.009    103.168      0.000         0.902     0.938
ik            -0.6186      0.023    -26.982      0.000        -0.665    -0.572
==============================================================================
Omnibus:                       11.015   Durbin-Watson:                   0.205
Prob(Omnibus):                  0.004   Jarque-Bera (JB):               11.740
Skew:                           1.308   Prob(JB):                      0.00282
Kurtosis:                       3.176   Cond. No.                         24.1
==============================================================================
"""



lm3 = smf.ols(formula='lny~lnk', data=df).fit()
print(lm3.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    lny   R-squared:                       0.941
Model:                            OLS   Adj. R-squared:                  0.939
Method:                 Least Squares   F-statistic:                     618.6
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           1.56e-25
Time:                        15:19:30   Log-Likelihood:                 127.05
No. Observations:                  41   AIC:                            -250.1
Df Residuals:                      39   BIC:                            -246.7
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.7227      0.014    -52.825      0.000        -0.750    -0.695
lnk            0.3516      0.014     24.871      0.000         0.323     0.380
==============================================================================
Omnibus:                       17.730   Durbin-Watson:                   0.131
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               21.353
Skew:                           1.699   Prob(JB):                     2.31e-05
Kurtosis:                       3.978   Cond. No.                         15.6
==============================================================================
"""



lm4 = smf.ols(formula='lny~ik', data=df).fit()
print(lm4.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    lny   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     735.4
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           6.39e-27
Time:                        15:19:30   Log-Likelihood:                 130.41
No. Observations:                  41   AIC:                            -256.8
Df Residuals:                      39   BIC:                            -253.4
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.0322      0.013     -2.454      0.019        -0.059    -0.006
ik            -0.9150      0.034    -27.118      0.000        -0.983    -0.847
==============================================================================
Omnibus:                       14.181   Durbin-Watson:                   0.181
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               15.839
Skew:                           1.495   Prob(JB):                     0.000364
Kurtosis:                       3.574   Cond. No.                         24.1
==============================================================================
"""



###
# 3

# Making the scatterplot with Plotly, an excellent package for graphing. I've commented it out bc you need the Plotly api to use it. Plotly makes nice graphs for R as well, btw. 

#trace = go.Scatter(x = df.k, y = df.y, mode='markers')
#data = [trace]
#plot_url = py.plot(data, filename = 'solow')

# The scatterplot itself is at https://plot.ly/~reddlee/604  it looks like Solow's scatterplot from the journal article we're writing about.

# Creating dummies

df['ww2'] = np.zeros(len(df))
for i in range(len(df)):
    if df['year'][i] >= 1943 and df['year'][i] <= 1949:
        df['ww2'][i] = 1
print(df)


lmd = smf.ols(formula='y~lnk+ww2', data=df).fit()
print(lmd.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.200e+04
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           6.06e-59
Time:                        15:38:47   Log-Likelihood:                 229.81
No. Observations:                  41   AIC:                            -453.6
Df Residuals:                      38   BIC:                            -448.5
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.4477      0.001    394.604      0.000         0.445     0.450
lnk            0.2396      0.001    205.057      0.000         0.237     0.242
ww2            0.0190      0.000     49.481      0.000         0.018     0.020
==============================================================================
Omnibus:                        2.068   Durbin-Watson:                   0.900
Prob(Omnibus):                  0.356   Jarque-Bera (JB):                1.877
Skew:                           0.502   Prob(JB):                        0.391
Kurtosis:                       2.701   Cond. No.                         15.8
==============================================================================
"""


lmd2 = smf.ols(formula='y~ik+ww2', data=df).fit()
print(lmd2.summary())
"""
    OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.993
Model:                            OLS   Adj. R-squared:                  0.993
Method:                 Least Squares   F-statistic:                     2780.
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           6.36e-42
Time:                        15:38:47   Log-Likelihood:                 187.53
No. Observations:                  41   AIC:                            -369.1
Df Residuals:                      38   BIC:                            -363.9
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.9167      0.003    277.380      0.000         0.910     0.923
ik            -0.6185      0.008    -72.879      0.000        -0.636    -0.601
ww2            0.0169      0.001     15.705      0.000         0.015     0.019
==============================================================================
Omnibus:                       12.738   Durbin-Watson:                   0.420
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               12.841
Skew:                           1.259   Prob(JB):                      0.00163
Kurtosis:                       4.086   Cond. No.                         24.4
==============================================================================
"""


lmd3 = smf.ols(formula='lny~lnk+ww2', data=df).fit()
print(lmd3.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    lny   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.307e+04
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           2.46e-59
Time:                        15:38:47   Log-Likelihood:                 214.75
No. Observations:                  41   AIC:                            -423.5
Df Residuals:                      38   BIC:                            -418.4
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.7298      0.002   -445.468      0.000        -0.733    -0.727
lnk            0.3538      0.002    209.691      0.000         0.350     0.357
ww2            0.0288      0.001     51.970      0.000         0.028     0.030
==============================================================================
Omnibus:                        9.686   Durbin-Watson:                   0.800
Prob(Omnibus):                  0.008   Jarque-Bera (JB):                8.871
Skew:                           0.958   Prob(JB):                       0.0119
Kurtosis:                       4.232   Cond. No.                         15.8
==============================================================================
"""


lmd4 = smf.ols(formula='lny~ik+ww2', data=df).fit()
print(lmd4.summary())
"""
OLS Regression Results                            
==============================================================================
Dep. Variable:                    lny   R-squared:                       0.996
Model:                            OLS   Adj. R-squared:                  0.996
Method:                 Least Squares   F-statistic:                     5102.
Date:                Sun, 15 Nov 2015   Prob (F-statistic):           6.59e-47
Time:                        15:38:47   Log-Likelihood:                 183.87
No. Observations:                  41   AIC:                            -361.7
Df Residuals:                      38   BIC:                            -356.6
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.0367      0.004    -10.145      0.000        -0.044    -0.029
ik            -0.9148      0.009    -98.600      0.000        -0.934    -0.896
ww2            0.0257      0.001     21.859      0.000         0.023     0.028
==============================================================================
Omnibus:                       12.466   Durbin-Watson:                   0.487
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               12.442
Skew:                           1.227   Prob(JB):                      0.00199
Kurtosis:                       4.125   Cond. No.                         24.4
==============================================================================
"""

# Betas have changed very slightly and r-squareds have increased.



###
# 4 

# To test whether or not the relationship was fundamentally different after 1943, I'll run Solow's four tests on data pre-1943 and again for the data post-1943. I'll then find the difference between the coefficients and divide it by the standard error of the difference (= sqrt(se1^2 + se2^2)). This gives me the t-score of the difference in the coefficients against the null hypothesis that the difference is zero. To test whether or not the relationship is different, we can look up the t-scores in a t-table (two-sided) and find the probability that, if the null is true, we would measure a difference in coefficients as large or larger than what we measured. 

# Defining a function to perform what I said in the previous paragraph

def difftest(preww2, postww2):
    pre_beta = preww2.params[1]
    post_beta = postww2.params[1]
    beta_diff = pre_beta - post_beta
    se = np.sqrt(preww2.bse[1]**2 + postww2.bse[1]**2)
    t = beta_diff / se
    return t

# Creating two dataframes, pre and post war

dfpr = df[df.year < 1943]
dfpo = df[df.year > 1943]

# Running the models on each dataframe

lmpr = smf.ols(formula='y~lnk', data=dfpr).fit()
lmpr2 = smf.ols(formula='y~ik', data=dfpr).fit()
lmpr3 = smf.ols(formula='lny~lnk', data=dfpr).fit()
lmpr4 = smf.ols(formula='lny~ik', data=dfpr).fit()

lmpo = smf.ols(formula='y~lnk', data=dfpo).fit()
lmpo2 = smf.ols(formula='y~ik', data=dfpo).fit()
lmpo3 = smf.ols(formula='lny~lnk', data=dfpo).fit()
lmpo4 = smf.ols(formula='lny~ik', data=dfpo).fit()

# Finding the t-scores of the differences in our coefficients

print(difftest(lmpr, lmpo))
print(difftest(lmpr2, lmpo2))
print(difftest(lmpr3, lmpo3))
print(difftest(lmpr4, lmpo4)) 

"""
1.86667280186
-1.23240774582
3.42553112524
-2.54274886797
"""

# Given the above t-scores, especially the last two, I would tentatively reject the null hypothesis that the difference is zero.

df = pd.read_csv("/home/rudebeans/Downloads/corpus.csv")
print(df)



###
# 5

glm = glm(formula='accept~black + hisp + othrace + age + edlt10 + ed10_11 + ed13_15 + edgt15', data = df).fit()
print(glm.summary())
"""
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                 accept   No. Observations:                 1133
Model:                            GLM   Df Residuals:                     1124
Model Family:                Gaussian   Df Model:                            8
Link Function:               identity   Scale:                   0.23341365525
Method:                          IRLS   Log-Likelihood:                -778.91
Date:                Mon, 16 Nov 2015   Deviance:                       262.36
Time:                        12:26:03   Pearson chi2:                     262.
No. Iterations:                     3                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept      0.3444      0.067      5.143      0.000         0.213     0.476
black         -0.0019      0.053     -0.036      0.971        -0.107     0.103
hisp           0.0002      0.034      0.004      0.996        -0.066     0.066
othrace       -0.0334      0.142     -0.235      0.815        -0.312     0.246
age            0.0016      0.002      0.875      0.381        -0.002     0.005
edlt10        -0.0848      0.038     -2.206      0.027        -0.160    -0.009
ed10_11        0.0106      0.039      0.273      0.785        -0.066     0.087
ed13_15       -0.0446      0.046     -0.975      0.330        -0.134     0.045
edgt15        -0.0074      0.095     -0.078      0.938        -0.194     0.179
==============================================================================
Optimization terminated successfully.
         Current function value: 0.655622
         Iterations 5
"""
# The only coefficients in which we have any confidence are for the intercept (the probability that a white high school graduate will be accepted) and for the coefficient on edlt10 (the decrease in the probability of acceptance when years of education are decreased from 12 to less than 10 at given levels of other variables). The magnitude of the latter is also notable. 



###
# 6

pred = glm.predict()
df['pred'] = pred
print(df.describe())
"""
              pred  
count  1133.000000  
mean      0.369815  
std       0.038647  
min       0.266434  
25%       0.338076  
50%       0.384636  
75%       0.397502  
max       0.442065  
"""
# It appears that none of the predicted probabilities falls outside of (0,1)



###
# 7

logit_mod = logit(formula='accept~black + hisp + othrace + age + edlt10 + ed10_11 + ed13_15 + edgt15', data = df).fit()
print(logit_mod.summary())
"""
Logit Regression Results                           
==============================================================================
Dep. Variable:                 accept   No. Observations:                 1133
Model:                          Logit   Df Residuals:                     1124
Method:                           MLE   Df Model:                            8
Date:                Mon, 16 Nov 2015   Pseudo R-squ.:                0.004910
Time:                        12:15:46   Log-Likelihood:                -742.82
converged:                       True   LL-Null:                       -746.48
                                        LLR p-value:                    0.5015
==============================================================================
                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.6444      0.287     -2.242      0.025        -1.208    -0.081
black         -0.0086      0.228     -0.038      0.970        -0.455     0.438
hisp           0.0009      0.145      0.006      0.995        -0.284     0.285
othrace       -0.1470      0.626     -0.235      0.814        -1.374     1.080
age            0.0069      0.008      0.879      0.379        -0.009     0.022
edlt10        -0.3722      0.168     -2.217      0.027        -0.701    -0.043
ed10_11        0.0443      0.165      0.269      0.788        -0.279     0.367
ed13_15       -0.1910      0.197     -0.969      0.332        -0.577     0.195
edgt15        -0.0319      0.404     -0.079      0.937        -0.824     0.760
==============================================================================
Optimization terminated successfully.
         Current function value: 0.655633
         Iterations 4
"""

probit_mod = probit(formula='accept~black + hisp + othrace + age + edlt10 + ed10_11 + ed13_15 + edgt15', data = df).fit()
print(probit_mod.summary())
"""
                          Probit Regression Results                           
==============================================================================
Dep. Variable:                 accept   No. Observations:                 1133
Model:                         Probit   Df Residuals:                     1124
Method:                           MLE   Df Model:                            8
Date:                Mon, 16 Nov 2015   Pseudo R-squ.:                0.004894
Time:                        12:15:46   Log-Likelihood:                -742.83
converged:                       True   LL-Null:                       -746.48
                                        LLR p-value:                    0.5039
==============================================================================
                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -0.4000      0.178     -2.247      0.025        -0.749    -0.051
black         -0.0041      0.141     -0.029      0.977        -0.280     0.272
hisp           0.0016      0.090      0.018      0.985        -0.174     0.177
othrace       -0.0888      0.382     -0.233      0.816        -0.838     0.660
age            0.0042      0.005      0.866      0.386        -0.005     0.014
edlt10        -0.2281      0.103     -2.219      0.026        -0.429    -0.027
ed10_11        0.0275      0.102      0.269      0.788        -0.173     0.228
ed13_15       -0.1175      0.121     -0.968      0.333        -0.356     0.120
edgt15        -0.0189      0.251     -0.075      0.940        -0.510     0.472
==============================================================================
"""

# The coefficients of the logit and probit models resemble one another closely, as they should given the fact that the only difference is that logit uses the logistic distribution and probit uses the normal. Coefficients for logit/probit models can't be interpreted as simply as those for linear models, so it's no surprise the logit/probit coefficients don't match the LPM coefficients. We can however, interpret the sign and significance of logit and probit coefficients, and in this sense they match those of the LPM. I think the negative intercept on the logit/probit models indicates that the probability of a white high-school graduate being accepted is less than %50, which is also indicated by the LPM intercept. 



###
# 8

mfx_logit = logit_mod.get_margeff()
print(mfx_logit.summary())
"""
        Logit Marginal Effects       
=====================================
Dep. Variable:                 accept
Method:                          dydx
At:                           overall
==============================================================================
                dy/dx    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
black         -0.0020      0.053     -0.038      0.970        -0.105     0.101
hisp           0.0002      0.034      0.006      0.995        -0.066     0.066
othrace       -0.0340      0.145     -0.235      0.814        -0.318     0.250
age            0.0016      0.002      0.880      0.379        -0.002     0.005
edlt10        -0.0862      0.039     -2.234      0.025        -0.162    -0.011
ed10_11        0.0103      0.038      0.269      0.788        -0.064     0.085
ed13_15       -0.0442      0.046     -0.971      0.332        -0.134     0.045
edgt15        -0.0074      0.094     -0.079      0.937        -0.191     0.176
==============================================================================
"""

mfx_probit = probit_mod.get_margeff()
print(mfx_probit.summary())
"""
       Probit Marginal Effects       
=====================================
Dep. Variable:                 accept
Method:                          dydx
At:                           overall
==============================================================================
                dy/dx    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
black         -0.0015      0.053     -0.029      0.977        -0.105     0.102
hisp           0.0006      0.034      0.018      0.985        -0.065     0.067
othrace       -0.0334      0.143     -0.233      0.816        -0.315     0.248
age            0.0016      0.002      0.867      0.386        -0.002     0.005
edlt10        -0.0856      0.038     -2.234      0.026        -0.161    -0.010
ed10_11        0.0103      0.038      0.269      0.788        -0.065     0.086
ed13_15       -0.0441      0.046     -0.969      0.332        -0.133     0.045
edgt15        -0.0071      0.094     -0.075      0.940        -0.192     0.177
==============================================================================
"""
# These marginal effects are the average of the marginal effects. We can also estimate marginal effects at the mean, median, or other values of our x-variables. The marginal effects of the LPM are simply the coefficients reported above. Looking at the only signifant result (edlt10) we see that all three models report pretty much the same marginal effects, which makes sense. 


