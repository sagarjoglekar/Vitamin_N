# import arviz as az
import pandas as pd
import numpy as np
import scipy
import pymc3 as pm
import matplotlib.pyplot as plt

#df_grocery = pd.read_csv('year_ward_grocery.csv')
df_grocery = pd.read_csv('ward_grocery.csv')
df_grocery['female_perc'] = df_grocery.apply(lambda row: row['female'] / row['population'], axis=1)
df_diabetes = pd.read_csv('diabetes_estimates_osward_2016.csv', encoding='utf-8', header=0).dropna()
df_geo = pd.read_csv('london_pcd2geo_2015.csv', encoding='utf-8')
df_geo = df_geo[['osward','oslaua']]
df_geo = df_geo.drop_duplicates()

df = df_grocery.merge(df_diabetes, how='inner', left_on='area_id', right_on='osward')
df = df.merge(df_geo, how='inner', on='osward')

plt.figure(figsize=(8, 8))
plt.plot(df['energy_carb'], df['estimated_diabetes_prevalence'], 'bo')
plt.xlabel('energy_carb', size = 18)
plt.ylabel('estimated_diabetes_prevalence', size = 18)

X1=df['energy_carb'].values
X2=df['h_energy_nutrients_norm'].values
X3=df['avg_age'].values
X4=df['female_perc'].values
X5=df['num_transactions'].values
X6=df['people_per_sq_km'].values

X5 = np.array([np.log2(x) for x in X5])
X6 = np.array([np.log2(x) for x in X6])

Y=df['estimated_diabetes_prevalence'].values


oslaua2index = {}
i=0
for v in df['oslaua'].values:
    if v not in oslaua2index:
        oslaua2index[v]=i
        i += 1

df['oslaua_idx'] = df.apply(lambda row : oslaua2index[row['oslaua']], axis=1)
    
n_oslauas = n_counties = len(df['oslaua_idx'].unique())
oslaua_idx = df['oslaua_idx'].values

hierarchical_model = pm.Model()
with hierarchical_model:
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a', mu=0., sd=100)
    sigma_a = pm.HalfNormal('sigma_a', 5.)
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_oslauas)
    
    mu_b1 = pm.Normal('mu_b1', mu=0., sd=100)
    sigma_b1 = pm.HalfNormal('sigma_b1', 5.)
    b1 = pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_oslauas)
    
    mu_b2 = pm.Normal('mu_b2', mu=0., sd=100)
    sigma_b2 = pm.HalfNormal('sigma_b2', 5.)
    b2 = pm.Normal('b2', mu=mu_b2, sd=sigma_b2, shape=n_oslauas)
    
    mu_b3 = pm.Normal('mu_b3', mu=0., sigma=100)
    sigma_b3 = pm.HalfNormal('sigma_b3', 5.)
    b3 = pm.Normal('b3', mu=mu_b3, sd=sigma_b3, shape=n_oslauas)
    
    mu_b4 = pm.Normal('mu_b4', mu=0., sigma=100)
    sigma_b4 = pm.HalfNormal('sigma_b4', 5.)
    b4 = pm.Normal('b4', mu=mu_b4, sd=sigma_b4, shape=n_oslauas)
    
    mu_b5 = pm.Normal('mu_b5', mu=0., sigma=100)
    sigma_b5 = pm.HalfNormal('sigma_b5', 5.)
    b5 = pm.Normal('b5', mu=mu_b5, sd=sigma_b5, shape=n_oslauas)
    
    mu_b6 = pm.Normal('mu_b6', mu=0., sigma=100)
    sigma_b6 = pm.HalfNormal('sigma_b6', 5.)
    b6 = pm.Normal('b6', mu=mu_b6, sd=sigma_b6, shape=n_oslauas)
    
    # Model error
    eps = pm.HalfCauchy('eps', 5.)
    
    estimate = a[oslaua_idx] + b1[oslaua_idx]*X1 + b2[oslaua_idx]*X2 + b3[oslaua_idx]*X3 + b4[oslaua_idx]*X4 + b5[oslaua_idx]*X5 + b6[oslaua_idx]*X6

    # Likelihood (sampling distribution) of observations
    likelihood = pm.Normal('likelihood', mu=estimate, sd=eps, observed=Y)
    
with hierarchical_model:
    hierarchical_trace = pm.sample(10000, tune=10000, target_accept=.9)
    
ppc = pm.sample_posterior_predictive(hierarchical_trace, samples=10000, model=hierarchical_model)

np.asarray(ppc['likelihood']).shape
# print(az.r2_score(Y, ppc['likelihood']))
