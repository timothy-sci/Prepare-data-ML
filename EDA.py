#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# The objective is to investigate which features and how they influence purchase price

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

