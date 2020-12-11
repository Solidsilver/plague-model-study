#!/usr/bin/env python
# coding: utf-8

# ## Lynch-Oster plague model

# #### Import packages

# In[10]:


# get_ipython().system(u'pip install pymc')
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from pymc.Matplot import plot
import scipy.stats


# #### Import observed data

# In[11]:


barcelona_1490 = np.array([1,0,1,1,0,1,5,3,1,0,1,1,2,3,5,0,6,3,6,3,8,1,5,2,1,1,2,2,2,5,7,12,4,3,5,3,8,5,8,8,6,12,11,22,15,14,24,14,15,20,20,13,11,25,28,30,24,28,42,24,32,24,27,31,34,33,29,31,38,40,42,38,53,44,66,52,53,56,63,49,60,57,65,55,55,47,67,62,65,57,47,46,62,54,52,48,49,64,46,67,52,50,56,46,41,38,36,39,31,32,41,25,32,35,36,36,33,26,42,31,19,27,23,22,15,24,32,19,10,16,12,15,14,13,12,13,12,6,12,15,5,9,3,5,12,6,7,3,3,3,3,2,3,3,0,3,2,3,3,1,1,4,2,3,0,2,3,2,0,1,1,4,1,2,2,1,1,2,0,1,1,2])
barcelona_pop = 25000.0

malta_1813 = np.array([1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 2, 8, 10, 6, 6, 4, 7, 4, 5, 5, 7, 16, 10, 12, 18, 8, 13, 19, 16, 24, 19, 28, 24, 32, 27, 33, 33, 36, 19, 19, 23, 28, 21, 18, 24, 32, 29, 33, 23, 34, 39, 37, 44, 47, 53, 42, 44, 53, 43, 49, 56, 55, 51, 52, 56, 63, 59, 55, 65, 67, 36, 50, 41, 43, 41, 55, 48, 45, 44, 53, 47, 64, 49, 53, 63, 58, 50, 48, 27, 47, 43, 35, 37, 24, 26, 28, 26, 31, 31, 29, 34, 28, 32, 38, 27, 32, 32, 34, 33, 34, 31, 38, 27, 25, 29, 28, 26, 37, 33, 32, 28, 34, 34, 38, 29, 30, 24, 20, 23, 17, 17, 30, 23, 12, 20, 24, 21, 12, 12, 16, 11, 11, 19, 12, 15, 14, 16, 12, 16, 15, 7, 6, 14, 13, 9, 8, 5, 3, 5, 4, 3, 5, 5, 5, 4, 7, 10, 2, 4, 7, 6, 2, 3, 5, 1, 5, 3, 1, 1, 3, 2, 3, 3, 1, 3, 0, 2, 0, 2, 2, 2, 1, 1, 3, 4, 2, 4, 1, 2, 1, 2, 1, 4, 1, 1])
malta_pop = 97000.0

florence_1400 = np.array([8, 19, 9, 7, 12, 10, 5, 16, 11, 16, 19, 22, 18, 16, 29, 30, 28, 29, 30, 28, 28, 28, 32, 38, 28, 46, 44, 44, 49, 49, 47, 52, 44, 44, 39, 57, 62, 67, 70, 64, 91, 87, 102, 103, 74, 81, 77, 73, 70, 82, 109, 98, 106, 104, 141, 147, 150, 161, 200, 161, 153, 147, 150, 173, 170, 204, 189, 176, 180, 180, 169, 137, 155, 170, 138, 136, 129, 152, 135, 118, 158, 125, 148, 153, 134, 126, 108, 121, 124, 101, 113, 70, 73, 73, 63, 76, 51, 84, 65, 76, 57, 55, 44, 47, 44, 32, 36, 27, 35, 37, 29, 35, 38, 36, 20, 51, 30, 42, 29, 27, 27, 23, 34, 22, 12, 21, 28, 18, 12, 18, 18, 21, 14, 28, 23, 20, 16, 15, 17, 23, 15, 13, 15, 8, 15, 12, 22, 5, 11, 8, 7, 13, 5, 10, 6, 12, 4, 13, 11, 10, 3, 5, 11, 5, 7, 5, 6, 3, 11, 5, 5, 11, 6, 5, 8, 6, 6, 6, 5, 6])
florence_pop = 60000.0

mortality_data = malta_1813
pop_size = malta_pop
t = len(mortality_data)


# #### Set up the model

# In[12]:


__all__ = ['mortality_data','sigma', 'beta', 'mu', 'v', 'phi', 'r'
           'S_0', 'E_0', 'I_0', 'R_0', 'D_0', 'D', 'D_h',
           'Rt_0', 'Rc_0', 'Ft_0', 'Fc_0',
           'mortality', 'pop_size', 'sus_frac', 'mortality_sim']

#Parameters
sigma = pm.Uniform('sigma', 1e-9, 1.0, value = 0.0015) # Estimated Parameter
beta = pm.Uniform('beta', 1e-9, 1.0, value = 0.0001)
mu = pm.Uniform('mu', 1e-9, 1.0, value = 0.0000001)
# mu = 0
v = pm.Uniform('v', 0.1, 4.0, value = 0.167)
r = pm.Uniform('r', 1e-9, 1.0, value = 0.1)
# phi = 1.0 - r
phi = pm.Uniform('phi', 1e-9, value = 0.1667)

beta_r = pm.Uniform('beta_r', 0.1, 1.0, value=0.2)
# k_r = pm.Uniform('k_r', 1e-9, 1.0, value=0.01)
# gamma = 0.1
# delta = 1.0 - gamma
gamma = pm.Uniform('gamma', 1e-9, 1.0, value = 0.1)
alpha = pm.Uniform('alpha', 1e-9, 1.0, value = 0.5)
delta = pm.Uniform('delta', 1e-9, 1.0, value = 0.0033)

beta_f = pm.Uniform('beta_f', 10.0, 100.0, value = 40.0)
# k_f = pm.Uniform('k_f', 0.25, 10., value = 1.0)
rho = pm.Uniform('rho', 1e-9, 1.0, value = 0.025)
# rho = 0.2
lamb = pm.Uniform('lamb', 1e-9, 1.0, value = 0.5)


sus_frac = pm.Uniform('sus_frac', 1e-9, 1.0, value = 0.14)
# pop_size = florence_pop

#Intial conditions
# Rats
Rt_0 = pop_size*sus_frac
Rc_0 = pm.Uniform('Rc_0', 1., 15., value=15.)
# Fleas
Ft_0 = 6. * Rt_0
Fc_0 = 1.
# Humans
S_0 = pop_size
E_0 = 1.0
I_0 = 1.5
R_0 = 0.0
D_0 = 1.0

#SEIRD model for pneumonic plague
@pm.deterministic
def SIRD(Rt_0=Rt_0, Rc_0=Rc_0,
        Ft_0=Ft_0, Fc_0=Fc_0,
        S_0=S_0, E_0=E_0, I_0=I_0, R_0=R_0, D_0=D_0,
        sigma=sigma, beta=beta, mu=mu, v=v, phi=phi, r=r,
        beta_r=beta_r, gamma=gamma, alpha=alpha, delta=delta,
        beta_f=beta_f, rho=rho, lamb=lamb):

    # Rat vars
    K_r = Rt_0 * 1.5 #carying capacity for rats
    R_t = np.zeros(t)
    R_c = np.zeros(t)

    # Flea vars
    F_t = np.zeros(t)
    F_c = np.zeros(t)

    # Human vars
    S = np.zeros(t)
    E = np.zeros(t)
    I = np.zeros(t)
    R = np.zeros(t)
    D = np.zeros(t)

    # Initial Conditions
    R_t[0] = Rt_0
    R_c[0] = Rc_0

    F_t[0] = Ft_0
    F_c[0] = Fc_0 

    S[0] = S_0
    E[0] = E_0
    I[0] = I_0
    R[0] = R_0
    D[0] = D_0    
    
    for i in range(1, t):

        K_f = 6.0 * R_t[i-1] #carying capacity for fleas
        intrinsic_flux_fleas = beta_r * F_t[i-1] * (K_f - F_t[i-1]) / K_f
        newly_infected_fleas = lamb * (R_c[i-1]/R_t[i-1]) * (F_t[i-1] - F_c[i-1])
        
        F_t[i] = F_t[i-1] + intrinsic_flux_fleas - rho * F_t[i-1]
        F_c[i] = F_c[i-1] + newly_infected_fleas - rho * F_c[i-1]
    
        intrinsic_flux_rats = beta_r * R_t[i-1] * (K_r - R_t[i-1]) / K_r
        intrinsic_flux_con_rats = beta_r * R_t[i-1] * R_c[i-1] / K_r
        newly_infected_rats = alpha * (F_c[i-1]/F_t[i-1]) * (R_t[i-1] - R_c[i-1])
        recovered_rats = gamma * R_c[i-1]
        plague_killed_rats = delta * R_c[i-1]
        
        R_t[i] = R_t[i-1] + intrinsic_flux_rats - plague_killed_rats
        R_c[i] = R_c[i-1] + newly_infected_rats - intrinsic_flux_con_rats - plague_killed_rats - recovered_rats

        total_pop_h = S[i-1] + E[i-1] + I[i-1] + R[i-1]
        intrinsic_birth_h = beta * (S[i-1] + R[i-1])
        exposed_from_flea = sigma * S[i-1] * F_c[i-1] / F_t[i-1]
        newly_infected_h = v * E[i-1]
        newly_killed_bub_h = phi * I[i-1]
        recovered_h = r * I[i-1]

        S[i] = S[i-1] + intrinsic_birth_h - exposed_from_flea - mu * S[i-1]
        E[i] = E[i-1] + exposed_from_flea - newly_infected_h
        I[i] = I[i-1] + newly_infected_h - newly_killed_bub_h - recovered_h
        R[i] = R[i-1] + recovered_h - mu * R[i-1]
        D[i] = newly_killed_bub_h + mu * total_pop_h

    return R_t, R_c, F_t, F_c, S, E, I, R, D

D = pm.Lambda('D', lambda SIRD=SIRD: SIRD[8])

#Likelihood
mortality = pm.Poisson('mortality', mu=D, value=mortality_data, observed=True)
mortality_sim = pm.Poisson('mortality_sim', mu=D)


# In[ ]:





# #### Fit the model

# In[13]:


if __name__ == '__main__':
    vars = [mortality_data, sigma, beta, mu, v, phi, r,
            beta_r, gamma, alpha, delta, beta_f, rho, lamb,
            S_0, E_0, I_0, R_0, D_0, D, Rc_0, mortality, 
            pop_size, sus_frac, mortality_sim]
    
    mc = pm.MCMC(vars, db='pickle', dbname='newratfle')
    mc.use_step_method(pm.AdaptiveMetropolis, [sigma, beta, v, r, mu, phi, beta_r, alpha, gamma, delta, beta_f, rho, lamb, sus_frac, Rc_0])
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.db.close()


# #### Output summary

# In[14]:


pm.gelman_rubin(mc)


# In[15]:


mc.summary()


# In[ ]:


M = pm.MAP(mc)
M.fit()
print("BIC:", M.BIC)
print("AIC:", M.AIC)


# #### Plot the posteriors

# In[ ]:


# get_ipython().magic(u'matplotlib inline')
plot(mc)


# #### Plot the fit

# In[ ]:


# get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(10,10))
plt.title('Florence 1400')
plt.xlabel('Day')
plt.ylabel('Deaths')
plt.plot(mortality_data, 'o', mec='black', color='black', label='Observed deaths')
plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='New Model (mean)')
y_min = mortality_sim.stats()['quantiles'][2.5]
y_max = mortality_sim.stats()['quantiles'][97.5]
plt.fill_between(range(0,len(mortality_data)), y_min, y_max, color='r', alpha=0.3, label='New Model (95% CI)')
plt.legend()

