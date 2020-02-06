import pandas as pd
import numpy as np
from sklearn import preprocessing
import pymc3 as pm

print('getting data')
X_dims = ['energy_carb', 'h_energy_nutrients_norm', 'avg_age', 'female_perc', 'num_transactions', 'people_per_sq_km']
Y_dims = ['estimated_diabetes_prevalence']

df_grocery = pd.read_csv('ward_grocery.csv')
df_grocery['female_perc'] = df_grocery.apply(lambda row: row['female'] / row['population'], axis=1)
df_diabetes = pd.read_csv('diabetes_estimates_osward_2016.csv', encoding='utf-8', header=0).dropna()

df = df_grocery.merge(df_diabetes, how='inner', left_on='area_id', right_on='osward')

# Create the features and response
#X = df.loc[:, ['Intercept', dim1]]
X = np.array(df[X_dims].values)
Y = np.array(df[Y_dims].values)

# rescaling data
min_max_scaler = preprocessing.MinMaxScaler() #StandardScaler() #MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y = min_max_scaler.fit_transform(Y)

X_pm = X.transpose()
Y_pm = Y.transpose()[0]

print('building model')
basic_model = pm.Model()
num_datapoints = 100
num_samples = 100
with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=len(X_pm))
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    #mu = alpha + beta[0]*X_pm[0] + beta[1]*X_pm[1]
    beta_terms = [beta[i] * X_pm[i][0:num_datapoints] for i in range(0,len(X_pm))]
    mu = alpha + sum(beta_terms)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y[0:num_datapoints])

print('sampling')
with basic_model:
    # Sampler
    step = pm.NUTS()

    # Posterior distribution
    trace = pm.sample(100, step)