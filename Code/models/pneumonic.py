import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from pymc.Matplot import plot
import scipy.stats


# Import data

barcelona_1490 = np.array([1,0,1,1,0,1,5,3,1,0,1,1,2,3,5,0,6,3,6,3,8,1,5,2,1,1,2,2,2,5,7,12,4,3,5,3,8,5,8,8,6,12,11,22,15,14,24,14,15,20,20,13,11,25,28,30,24,28,42,24,32,24,27,31,34,33,29,31,38,40,42,38,53,44,66,52,53,56,63,49,60,57,65,55,55,47,67,62,65,57,47,46,62,54,52,48,49,64,46,67,52,50,56,46,41,38,36,39,31,32,41,25,32,35,36,36,33,26,42,31,19,27,23,22,15,24,32,19,10,16,12,15,14,13,12,13,12,6,12,15,5,9,3,5,12,6,7,3,3,3,3,2,3,3,0,3,2,3,3,1,1,4,2,3,0,2,3,2,0,1,1,4,1,2,2,1,1,2,0,1,1,2])
mortality_data = barcelona_1490
t = len(mortality_data)


# Set up the model

__all__ = ['mortality_data','beta', 'gamma', 
           'S_0', 'I_0', 'D_0', 'D', 'D_h', 'mortality', 
           'pop_size', 'sus_frac', 'mortality_sim']

#Parameters
beta = pm.Uniform('beta', 1e-9, 1.0)
gamma = .4

sus_frac = pm.Uniform('sus_frac', 1e-9, 1.)
pop_size = 25000.

#Intial conditions
S_0 = pop_size*sus_frac
I_0 = pm.Uniform('I_0', 1., 10.)
D_0 = 1.

#SID model for pneumonic plague
@pm.deterministic
def SID(S_0=S_0, I_0=I_0, D_0=D_0, beta=beta, gamma=gamma):
    S = np.zeros(t)
    I = np.zeros(t)
    D = np.zeros(t)

    S[0] = S_0
    I[0] = I_0
    D[0] = D_0
    
    for i in range(1, t):
        new_infected = min(S[i-1], beta*S[i-1]*I[i-1]/(S[i-1]+I[i-1]))
        new_dead = min(I[i-1], gamma*I[i-1])
        
        S[i] = S[i-1] - new_infected
        I[i] = I[i-1] + new_infected - new_dead
        D[i] = new_dead

    return S, I, D

D_h = pm.Lambda('D_h', lambda SID=SID: SID[2])

#Likelihood
mortality = pm.Poisson('mortality', mu=D_h, value=mortality_data, observed=True)
mortality_sim = pm.Poisson('mortality_sim', mu=D_h)

# Fit the model

if __name__ == '__main__':
    vars = [mortality_data, beta, gamma, 
            S_0,I_0, D_0, D_h, mortality, 
            pop_size, sus_frac, mortality_sim]
    
    mc = pm.MCMC(vars, db='pickle', dbname='pneu')
    mc.use_step_method(pm.AdaptiveMetropolis, [beta, sus_frac, I_0])
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.db.close()


# Output summary
pm.gelman_rubin(mc)
mc.summary()

M = pm.MAP(mc)
M.fit()
M.BIC

# Plot posteriors
plot(mc)

# Plot the fit
plt.figure(figsize=(10,10))
plt.title('Barcelona 1490')
plt.xlabel('Day')
plt.ylabel('Deaths')
plt.plot(mortality_data, 'o', mec='black', color='black', label='Observed deaths')
plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='PPP (mean)')
y_min = mortality_sim.stats()['quantiles'][2.5]
y_max = mortality_sim.stats()['quantiles'][97.5]
plt.fill_between(range(0,len(mortality_data)), y_min, y_max, color='r', alpha=0.3, label='PPP (95% CI)')
plt.legend()
