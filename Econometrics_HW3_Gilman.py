#########################################
# Econometrics, HW3                     #
# Rudy Gilman                           #
# October 7, 2015                       #
#########################################


import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import sklearn, scipy.stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

loc = '/home/rudebeans/Downloads/1111531048_364888/'


#######################################################
### C1

bw = pd.read_stata(loc+"BWGHT.DTA")

#i. Positive. More money means better care for mom

#ii. Yes, they're probably negatively correlated. Poor people seem to smoke more.

lmc = smf.ols(formula='bwght ~ cigs', data = bw).fit()

print(lmc.params)
print(lmc.summary())


#I've copied and pasted from the terminal. If you actually need this written out explicitly as an equation please let me know and I'll send another copy:

"""
Without faminc:
Intercept 119.771900
cigs       -0.513772
N =         1388
R2 =         .023
"""
lm = smf.ols(formula='bwght ~ cigs + faminc', data = bw).fit()
    
"""
With faminc
Intercept    116.974130
cigs          -0.463408
faminc         0.092765
No. Observations:1388
R-squared:     0.030
"""

#iii. Faminc does not significantly alter the effects of cigs on weight, though it does offer slightly more explanatory power to our model. Turns out faminc and cigs aren't very correlated



###################################################
### C2

hp = pd.read_stata(loc+"hprice1.dta")

lm = smf.ols(formula='price~sqrft+bdrms', data=hp).fit()

#i. 

"""
Intercept   -19.314996
sqrft         0.128436
bdrms        15.198191
r2             .632
N            88
"""
print(lm.summary())
#ii. $15,198

#iii. $33,179

#iv. About 63%

#v. $353,544
print(0.128436*(2438)+15.198191*4)

#vi. Residual 53,544. Buyer got a deal



################################################################
### C3

cs = pd.read_stata(loc+"CEOSAL2.DTA")

#i. See below

cs['lnsales'] = np.log(cs['sales'])
cs['lnmktval'] = np.log(cs['mktval'])
cs['lnsalary'] = np.log(cs['salary'])

lm = smf.ols(formula='lnsalary~lnsales+lnmktval', data=cs).fit()

"""
Intercept    4.620917
lnsales      0.162128
lnmktval     0.106708
r2            .299
N          177
"""

#ii Profits can't be included bc many of them are negative. Adding profits doesn't do anything for the model, anyways. Together, the model's variables explains 30% of variation in salary--not much.

lm = smf.ols(formula='lnsalary~lnsales+lnmktval+profits', data=cs).fit()

"""
Intercept    4.686924
lnsales      0.161368
lnmktval     0.097529
profits      0.000036
r2            .299
N          177
"""

#iii 1.1685%
lm = smf.ols(formula='lnsalary~lnsales+lnmktval+profits+ceoten', data=cs).fit()

#iv 0.77689759. These variables are highly correlated. This isn't a problem in terms of biasing our model, but it may increase variance of our estimators.

cm = np.corrcoef(cs['lnmktval'], cs['profits'])



######################################################
###C4

at = pd.read_stata(loc+"attend.dta")

#i. atndrte: (6.25, 100.0, 81.709558823529406) 
#   priGPA: (0.85699999, 3.9300001, 2.5867760153377759)
#   ACT: (13, 32, 22.51029411764706)

print(min(at['ACT']), max(at['ACT']), np.mean(at['ACT']))

#ii. Even with a GPA and ACT of 0 a student will be expected to attend 76% of the time. No student has these stats, though, so that's not really useful info.

lm = smf.ols(formula='atndrte~priGPA+ACT', data=at).fit()

"""
Intercept    75.700405
priGPA       17.260591
ACT          -1.716553
R2             .291
N           680
"""

#ii It's surprising that ACT has a negative coefficient bc we expect smart people to go to class more. GPA coefficient is unsurprising as we would expect diligent students to both go to class more and to study more.

#iii 104.3. Rate's can't be higher than 100%, but I guess that's a limitation of using a linear model. Yes, a student has these stats--their attendence was 87.5

ans = at[at['priGPA'] == 3.65]
print(ans)

#iv Difference is 25.9



###########################################################
### C5. Coefficients are identical, as expected

wg1 = pd.read_stata(loc+"WAGE1.DTA")

lm1 = smf.ols(formula='educ~exper+tenure', data=wg1).fit()

"""
Intercept    13.574964
exper        -0.073785
tenure        0.047680
"""

fit = lm1.predict()

wg1['r1'] = wg1.educ - fit

lm2 = smf.ols(formula='lwage~r1', data=wg1).fit()

"""
Intercept    1.623268
r1           0.092029
r2            .207
N           526
"""

lm3 = smf.ols(formula='lwage~educ+exper+tenure', data=wg1).fit()

"""
Intercept    0.284360
educ         0.092029
exper        0.004121
tenure       0.022067

"""



##############################################################
### C6

wg = pd.read_stata(loc+"WAGE2.DTA")

#i. Intercept    53.687154
#   educ          3.533829

lm = smf.ols(formula='IQ~educ', data=wg).fit()

#ii. Intercept    5.973062
#    educ         0.059839

lnwg = np.log(wg['wage'])

lm = smf.ols(formula='lnwg~educ', data=wg).fit()

#iii. Intercept    5.658288
#     educ         0.039120
#     IQ           0.005863

lm = smf.ols(formula='lnwg~educ+IQ', data=wg).fit()

#iv. ans = 0.059838839427, see below

ans = 0.039120+0.005863*3.533829



##############################################################
### C7

#i. Coefficients are what we'd expect. Spending increases pass rate, lnchprg (poverty) decreases it

mea = pd.read_stata(loc+"MEAP93.DTA")

lnexpend = np.log(mea['expend'])

lm = smf.ols(formula='math10~lnexpend+lnchprg', data=mea).fit()

"""
Intercept   -20.360816
lnexpend      6.229698
lnchprg      -0.304585
r2             .214
N           935
"""

#ii. The intercept isn't useful. We'd never spend that little on a student (1 dollar). 

lm = smf.ols(formula='math10~lnexpend', data=mea).fit()

#iii. The effect is much larger

"""
Intercept   -69.341161
lnexpend     11.164402
"""

#iv. -0.19270422. This makes sense. Poorer schools spend less and have more poor students needing lunch.

cm = np.corrcoef(lnexpend, mea['lnchprg'])

#v. The model without lnchprg fails to include the effects of poverty, thus overestimating the effects of spending.


#############################################################
### C8

dis = pd.read_stata(loc+"discrim.dta")

#i.(avg: 0.11348654413573026, 47053.789731051344, 
#   std: 0.18219332799293844, 13163.164612055622)
# prpblck is a percentage, income is in dollars

print(np.mean(dis['prpblck']), np.mean(dis['income']), np.std(dis['prpblck']), np.std(dis['income']))

#ii. An all-black population spends 11 cents more than an all white population. That's not a lot, but still significant

lm = smf.ols(formula='psoda~prpblck+income', data=dis).fit()

"""
Intercept    0.956320
prpblck      0.114988
income       0.000002
r2            .064
N          401
"""

#iii. Disc effect is lower. Income and prpblck are negatively correlated

lm = smf.ols(formula='psoda~prpblck', data=dis).fit()

"""
Intercept    1.037399
prpblck      0.064927
r2            .018
N           401
"""

#iv. 2.4%

dis['lnsoda'] = np.log(dis.psoda)
dis['lnincome'] = np.log(dis.income)

lm = smf.ols(formula='lnsoda~prpblck+lnincome', data=dis).fit()

"""
Intercept   -0.793768
prpblck      0.121580
lnincome     0.076511
r2            .068
N           401
"""

#v. prpblck coef falls to 0.072807

lm = smf.ols(formula='lnsoda~prpblck+lnincome+prppov', data=dis).fit()

"""
Intercept   -1.463332
prpblck      0.072807
lnincome     0.136955
prppov       0.380360
r2            .087
N          401
"""

#vi. -.84. It's what we would expect. Poverty is calculated using income.

cm = np.ma.corrcoef(dis['lincome'], dis['prppov'])

#vii Argument doesn't hold water. We're trying to thoroughly control for income, it's fine that they're correlated.



###############################################################
### Extra question

"""
Including an irrelevent variable should be fine if it's uncorrelated with the other (relevent) variables in the model. If it's correlated with other relevent variables, however, it will increase the variance of our estimators. Estimators remain unbiased in both cases.
"""








