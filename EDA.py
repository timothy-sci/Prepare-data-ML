#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# The objective is to investigate which features and how they influence purchase price
# carried over from the production side.

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 
        'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width',
        'height', 'curb-weight', 'engine-type', 'num-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

data = pd.read_csv('car_price.csv', names=cols)

# Replace bad data entries
data = data.replace('?', np.NaN)

#print(data.shape)
# (205, 26)


# Substitute nan values with average of a column
norm_loss_avg = data['normalized-losses'].astype('float').mean()
data['normalized-losses'].replace(np.NaN, norm_loss_avg, inplace=True)

bore_avg = data['bore'].astype('float').mean()
data['bore'].replace(np.NaN, bore_avg, inplace=True)


rpm_avg = data['peak-rpm'].astype('float').mean()
data['peak-rpm'].replace(np.NaN, rpm_avg, inplace=True)

# Drop rows of nan horsepower entries
data.dropna(subset = ['horsepower','num-of-doors','stroke','price'], axis=0, inplace=True)

# Convert to proper data types

data[['bore','stroke','peak-rpm','price']] = data[['bore','stroke','peak-rpm','price']].astype('float')
data[['normalized-losses','horsepower']] = data[['normalized-losses','horsepower']].astype('int')
print(data.dtypes)

#--------------------------------------------------------------------------
# Relationship between numerical variables --------------------------------
#--------------------------------------------------------------------------

# We look at Pearson correlation:
# Independent variables: Compression-ratio, horsepower, engine-size
# Dependent variable: Price

selected_var = ['engine-size',
        'compression-ratio','horsepower','price']
print(data[selected_var].corr(method='pearson'))

#---------------------------------------------------------------------------
# Output:
#                     engine-size  compression-ratio  horsepower  price
# engine-size             1             0.025257       0.845325   0.888778
# compression-ratio     0.025257            1         -0.203818   0.074483
# horsepower            0.845325       -0.203818          1       0.812453
# price                 0.888778        0.074483       0.812453      1
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Interpretation:
# Correlation between independent and dependent variables in the order:
# strong >> weak
# engine-size > horsepower > compression-ratio
# Because these correlations are positive, this means the price increase as
# engine-size increases. Because engine-size is postive, strongly correlated
# to horsepower, meaning to achieve lot of horsepower requires a bigger
# engine. In turn compression-ratio must decrease as horserpower increase.
# This is illustrated in the figures below:
# Data points are shown as scatter plot and estimation of linear regression
# between independent and dependent variables by a straight line
#
# Engine-size and horsepower seems like a good predictor of price
# Compression-ratio not a good predicto because of weak linear relationship
# with price. This means when we build ML algorithm to predict purchase
# price, compression-ratio can be dropped during dimensionality-reduction.
#----------------------------------------------------------------------------

plt.figure(1)
sns.regplot(x='engine-size',y='price', data = data)
plt.ylim(0,)

plt.figure(2)
sns.regplot(x='horsepower',y='price', data=data)
plt.ylim(0,)

plt.figure(3)
sns.regplot(x='compression-ratio',y='price', data=data)
plt.ylim(0,)

plt.show()

#-----------------------------------------------------------------------------
# Relationship with categorical variables ------------------------------------
#-----------------------------------------------------------------------------

print(data.describe(include=['object']))

# Look for average prices within categories.
# To do this, use groupby
# From this we can look at a specific car make and get average price
# grouped by driven wheels and body style
first_group = data[['make','drive-wheels','body-style','price']]
result1 = first_group.groupby(['make','drive-wheels','body-style'], as_index=False).mean()
print(result1)

#------------------------------------------------------------------------------
# Highlighting features/predictors useful for prediction with ML --------------
#------------------------------------------------------------------------------

# For builing ML algorithm, we can at this stage focus on the concepts:
# correlation and causation.
# We will look at the p-value, which defines the probability that the
# correlation between two variables is statistically significant.

#-------------------------------------------------------------------------------
# Interpretation of p-value:
# p-value < 0.001: strong evidence that correlation is significant
# p-value < 0.05: moderate evidence that correlation is significant
# p-value < 0.1: weak evidence that correlation is significant
# p-value > 0.1: no evidence that correlation is significant
#-------------------------------------------------------------------------------

pearson_coef, p_value = stats.pearsonr(data['engine-size'],data['price'])
print('Pearson coeff is {}'.format(pearson_coef))
print('p-value is {}'.format(p_value))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#Output:
# Pearson coeff is 0.88878
# p-value is 1.2525079e-66

pearson_coef, p_value = stats.pearsonr(data['wheel-base'],data['price'])
print('Pearson coeff is {}'.format(pearson_coef))
print('p-value is {}'.format(p_value))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#Output:
# Pearson coeff is 0.584950
# p-value is 4.164297e-19

pearson_coef, p_value = stats.pearsonr(data['length'],data['price'])
print('Pearson coeff is {}'.format(pearson_coef))
print('p-value is {}'.format(p_value))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#Output:
# Pearson coeff is 0.69592
# p-value is 2.809266e-29

pearson_coef, p_value = stats.pearsonr(data['width'],data['price'])
print('Pearson coeff is {}'.format(pearson_coef))
print('p-value is {}'.format(p_value))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#Output:
# Pearson coeff is 0.754648
# p-value is 8.440099e-37

pearson_coef, p_value = stats.pearsonr(data['curb-weight'],data['price'])
print('Pearson coeff is {}'.format(pearson_coef))
print('p-value is {}'.format(p_value))

#Output:
# Pearson coeff is 0.835367
# p-value is 1.587586e-51

#-------------------------------------------------------------------------
# Interpretation:
# Pearson correlation coefficients show that some of these variables
# have strong correlation and others moderate correlation with price,
# however the corresponding p-values shows that the correlations are
# significant, which means when building a ML algorithim for predicting 
# price, the features that have a siginificant effect on the price
# are: length, width, curb-weight, engine-size, horsepower, mpg, wheel-base,
# bore.
#
# Categorical variable drive-wheels also has an effect  on the price.
#--------------------------------------------------------------------------

print(data.info())

drop_cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 
        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 
        'engine-type', 'fuel-system', 'num-cylinders','compression-ratio',
        'peak-rpm', 'height']

data = data.drop(drop_cols, axis=1)

# Change data type

data['curb-weight'] = data['curb-weight'].astype(float)
data['engine-size'] = data['engine-size'].astype(float)
data['horsepower'] = data['horsepower'].astype(float)
data['city-mpg'] = data['city-mpg'].astype(float)
data['highway-mpg'] = data['highway-mpg'].astype(float)

X = data.drop(['price'],axis=1)
y = data['price']
print(X.dtypes)
# Prepare data for the model

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2, random_state=0)

# Feature scaling

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

#models = [ensemble.RandomForestRegressor(),
#        ensemble.GradientBoostingRegressor()]

print('~~~~~~~~~~ RandomForestRegressor ~~~~~~~~~~~~~~~~~~~~~~~')

# Define Random Forest Regression model

def RFR(train_X,train_y,test_X):
    rfrregressor = RandomForestRegressor(n_estimators=1000,
        min_samples_leaf=1)
    rfrfit = rfrregressor.fit(train_X, train_y)
    y_pred = rfrfit.predict(test_X)
    model_score = rfrfit.score(test_X,test_y)

    return y_pred

y_predicted = RFR(train_X,train_y,test_X)

# Evaluate the model

mae = metrics.mean_absolute_error(test_y,y_predicted)
mse = metrics.mean_squared_error(test_y,y_predicted)
rmse = np.sqrt(metrics.mean_squared_error(test_y,y_predicted))

print('Mean Absolute Error is {}'.format(mae))
print('Mean Squared Error is {}'.format(mse))
print('Root Mean Squared Error is {}'.format(rmse))


pred = pd.DataFrame.from_dict({'predicted':y_predicted, 'true':test_y})
pred['difference'] = pred.predicted - pred.true
pred['ratio'] = 1.0 - (pred.true/pred.predicted)
print(pred.sample(n=10).round(2))

print('~~~~~~~~~~ GradientBoostingRegressor ~~~~~')

# Define Random Forest Regression model

def GBR(train_X,train_y,test_X):
    gbrregressor = GradientBoostingRegressor(n_estimators=1000,
        max_depth=5,
        learning_rate=0.01)
    gbrfit = gbrregressor.fit(train_X, train_y)
    y_pred = gbrfit.predict(test_X)
    model_score = gbrfit.score(test_X,test_y)

    return y_pred

y_predicted = GBR(train_X,train_y,test_X)

# Evaluate the model

mae = metrics.mean_absolute_error(test_y,y_predicted)
mse = metrics.mean_squared_error(test_y,y_predicted)
rmse = np.sqrt(metrics.mean_squared_error(test_y,y_predicted))

print('Mean Absolute Error is {}'.format(mae))
print('Mean Squared Error is {}'.format(mse))
print('Root Mean Squared Error is {}'.format(rmse))


pred = pd.DataFrame.from_dict({'predicted':y_predicted, 'true':test_y})
pred['difference'] = pred.predicted - pred.true
pred['ratio'] = 1.0 - (pred.true/pred.predicted)
print(pred.sample(n=10).round(2))

# Output from predictions
'''
~~~~~~~~~~ RandomForestRegressor ~~~~~~~~~~~~~~~~~~~~~~~
Mean Absolute Error is 0.09436512410919157
Mean Squared Error is 0.014917654505934057
Root Mean Squared Error is 0.12213785042293014

     predicted  true  difference  ratio
166       9.29  9.16        0.13   0.01
120       8.78  8.74        0.04   0.00
158       8.98  8.97        0.00   0.00
135       9.56  9.65       -0.09  -0.01
165       9.24  9.14        0.10   0.01
64        9.15  9.33       -0.17  -0.02
191       9.54  9.50        0.05   0.00
190       9.14  9.21       -0.07  -0.01
138       8.90  8.54        0.36   0.04
193       9.41  9.42       -0.01  -0.00

~~~~~~~~~~ GradientBoostingRegressor ~~~~~
Mean Absolute Error is 0.09943324131671528
Mean Squared Error is 0.018526934798286727
Root Mean Squared Error is 0.1361136833616912

     predicted  true  difference  ratio
120       8.76  8.74        0.02   0.00
122       8.90  8.94       -0.04  -0.00
75        9.74  9.71        0.03   0.00
190       9.15  9.21       -0.05  -0.01
5         9.49  9.63       -0.14  -0.02
170       9.33  9.32        0.01   0.00
7         9.82  9.85       -0.03  -0.00
4         9.62  9.77       -0.15  -0.02
185       8.99  9.01       -0.02  -0.00
80        9.14  9.21       -0.06  -0.01
'''


