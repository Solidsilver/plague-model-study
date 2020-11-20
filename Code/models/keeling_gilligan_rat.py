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
__all__ = [
    'S_r0', 'I_r0', 'R_r0', 'D_r0',
    'S_h0', 'I_h0', 'R_h0','D_h0',
    'H_f', 'S_r', 'I_r', 'R_r', 'D_r',
    'S_h', 'I_h', 'R_h', 'D_h'
    'human_mortality', 'rat_mortality',
    'mortality_sim']

#Parameters
beta_r = pm.Uniform('beta_r', 1e-9, 1.0, value = .08)
gamma = 0.2
sigma = 0.1
beta_h = pm.Uniform('beta_h', 1e-9, .2, value = .2)

sus_frac = pm.Uniform('sus_frac', 1e-9, 1.0, value = .14)
pop_size = 25000.

H_f0 = 6.0
I_f0 = 1.*H_f0

#Initial conditions for rat population
S_r0 = pop_size*sus_frac
I_r0 = pm.Uniform('I_r0', 1., 15., value=15.)
R_r0 = 0.
D_r0 = 0.

#Initial conditions for human population
S_h0 = 25000.
I_h0 = 1.*1.5
R_h0 = 0.
D_h0 = 1.

searching = 3./S_r0

#SIRD model for rat-flea palgue
@pm.deterministic 
def SIRD(H_f0=H_f0, I_f0=I_f0,
         S_r0=S_r0, I_r0=I_r0, R_r0=R_r0, D_r0=D_r0, 
         S_h0=S_h0, I_h0=I_h0, R_h0=R_h0, D_h0=D_h0,
         beta_r=beta_r, gamma=gamma, sigma=sigma, 
         searching=searching, beta_h=beta_h):
    
    H_f = np.zeros(t)
    I_f = np.zeros(t)

    S_r = np.zeros(t)
    I_r = np.zeros(t)
    R_r = np.zeros(t)
    D_r = np.zeros(t)

    S_h = np.zeros(t)
    I_h = np.zeros(t)
    R_h = np.zeros(t)
    D_h = np.zeros(t)
   
    H_f[0] = H_f0
    I_f[0] = I_f0
    
    S_r[0] = S_r0
    I_r[0] = I_r0
    R_r[0] = R_r0
    D_r[0] = D_r0

    S_h[0] = S_h0
    I_h[0] = I_h0
    R_h[0] = R_h0
    D_h[0] = D_h0

    for i in range(1,t):
        if i == 1: #Referenced before assignment at i = 1
            infected_rat_deaths=D_h0
            N_r = S_r[i-1]+I_r[i-1]+R_r[i-1]
           
        #Fleas
        K_f = H_f0 #avg number of fleas per rat at carrying capacity
        if H_f[i-1]/K_f < 1.:
            flea_growth = .0084*(H_f[i-1]*(1.-(H_f[i-1]/K_f)))
        elif H_f[i-1]/K_f > 1.:
            flea_growth = -.0084*(H_f[i-1]*(1.-(H_f[i-1]/K_f)))
        else:
            flea_growth = 0.
           
        new_infectious = infected_rat_deaths*(H_f[i-1])
        starvation_deaths = 0.2*I_f[i-1]
        force_to_humans = min(I_f[i-1], I_f[i-1]*np.exp(-searching*N_r)) #number of fleas that find a human
        force_to_rats = I_f[i-1]-force_to_humans #number of fleas that find a rat
       
        H_f[i] = H_f[i-1] + flea_growth
        I_f[i] = I_f[i-1] + new_infectious - starvation_deaths
       
        #Rats
        N_r = S_r[i-1]+I_r[i-1]+R_r[i-1]

        new_infected_rats = min(S_r[i-1], beta_r*S_r[i-1]*force_to_rats/N_r)
        new_removed_rats = gamma*I_r[i-1]
        new_recovered_rats = sigma*new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats
       
        S_r[i] = S_r[i-1] - new_infected_rats 
        I_r[i] = I_r[i-1] + new_infected_rats - new_removed_rats 
        R_r[i] = R_r[i-1] + new_recovered_rats 
        D_r[i] = new_dead_rats
       
        #Humans
        N_h = S_h[i-1]+I_h[i-1]+R_h[i-1]
        
        new_infected_humans = min(S_h[i-1], beta_h*S_h[i-1]*force_to_humans/N_h)
        new_removed_humans = .1*I_h[i-1]
        new_recovered_humans = .4*new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans
        
        S_h[i] = S_h[i-1] - new_infected_humans
        I_h[i] = I_h[i-1] + new_infected_humans - new_removed_humans
        R_h[i] = R_h[i-1] + new_recovered_humans
        D_h[i] = new_dead_humans
       
    return H_f, I_f, S_r, I_r, R_r, D_r, I_h, D_h

D_h = pm.Lambda('D_h', lambda SIRD=SIRD: SIRD[7])

#Likelihood
mortality = pm.Poisson('mortality', mu=D_h, value=mortality_data, observed=True)
mortality_sim = pm.Poisson('mortality_sim', mu=D_h)


#Fit with mcmc
if __name__ == '__main__':
    vars = [beta_r, gamma, sigma, searching,
    H_f0, I_r0, R_r0, D_r0, S_h0,
    I_h0, D_h0, D_h, beta_h,
    pop_size, sus_frac, mortality,
    mortality_data, mortality_sim]
    
    mc = pm.MCMC(vars, db='pickle', dbname='rat')
    mc.use_step_method(pm.AdaptiveMetropolis, [beta_r,beta_h, sus_frac, I_r0])
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.sample(iter=180000, burn=80000, thin=10, verbose=0)
    mc.db.close()

# Output Summary
pm.gelman_rubin(mc)
mc.summary()

M = pm.MAP(mc)
M.fit()
M.BIC


# Plot the posteriors
plot(mc)

# Plot the fit
plt.figure(figsize=(10,10))
plt.title('Barcelona 1490')
plt.xlabel('Day')
plt.ylabel('Deaths')
plt.plot(mortality_data, 'o', mec='black', color='black', label='Observed deaths')
plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
y_min = mortality_sim.stats()['quantiles'][2.5]
y_max = mortality_sim.stats()['quantiles'][97.5]
plt.fill_between(range(0,len(mortality_data)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
plt.legend()
