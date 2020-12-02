import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import warnings

az.style.use('arviz-darkgrid')

with pm.Model() as model:
    mu = pm.Normal("mu", mu = 0, sigma = 1)
    obs = pm.Normal("obs", mu = mu, sigma = 1, observed = np.ransom.randn(100))

    model.basic_RVs

