#https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import os

# the dataset
data = pd.read_csv('G:\DataScienceProject\General\BMI.csv')
# creating dummies for gender
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Eng1'] = data['Height']**0.5
# the independent variables set
X = data[['Gender', 'Height', 'Weight', 'Eng1']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

print(vif_data)
#=========https://www.geeksforgeeks.org/box-cox-transformation-using-python/
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

original_data  = data['Height']

# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(original_data)

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# plotting the original data(non-normal) and
# fitted data (normal)
sns.distplot(original_data, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Non-Normal", color="green", ax=ax[0])

sns.distplot(fitted_data, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Normal", color="green", ax=ax[1])

# adding legends to the subplots
plt.legend(loc="upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

