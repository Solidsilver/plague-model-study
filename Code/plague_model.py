import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from pymc.Matplot import plot


class PlagueModel:

    def __init__(self, modelName, mort_data, pop_size, data_name = "generic_data", params = dict(), inputVars = [], mcmcVars = []):
        self.name = modelName
        self.params = params
        self.mcvars = mcmcVars
        self.vars = inputVars
        self.deaths = None
        self.mort_data = mort_data
        self.mortality = None
        self.mortality_sim = None
        self.mc = None
        self.t = len(mort_data)
        self.pop_size = pop_size
        self.data_name = data_name

    def model(self):
        """Define model here, then return relevent array of values
        Be sure to return the death data to fit as the first element"""
        pass

    def getName(self):
        return self.name

    def runMCMC(self, iterations = 180000, burn_in = 80000, thinning = 10, verbose = 0, samples = 3):
        self.deaths = pm.Lambda("deaths", lambda MOD=self.model: MOD[0])
        self.mortality = pm.Poisson('mortality', mu=self.deaths, value=self.mort_data, observed=True)
        self.mortality_sim = pm.Poisson('mortality_sim', mu=self.deaths)
        vars = self.vars
        vars.append(self.mort_data)
        vars.apppend(self.deaths)
        vars.append(self.mortality)
        vars.append(self.mortality_sim)

        mc = pm.MCMC(vars, db='pickle', dbname = self.name + "_db")
        mc.use_step_method(pm.AdaptiveMetropolis, self.mcvars)
        for i in range(samples):
            mc.sample(iter=iterations, burn=burn_in, thin=thinning, verbose=verbose)
            print("Done with iteration", i, "of", samples)
        mc.db.close()
        self.mc = mc

    def gelman_rubin(self):
        if self.mc is None:
            raise Exception("MC not initialized. First call runMCMC()")
        return pm.gelman_rubin(self.mc)
    
    def printMCSummary(self):
        if self.mc is None:
            raise Exception("MC not initialized. First call runMCMC()")
        return self.mc.summary()

    def checkFit(self):
        if self.mc is None:
            raise Exception("MC not initialized. First call runMCMC()")
        print("Mapping mc samples")
        M = pm.MAP(self.mc)
        print("Fitting map")
        return (M.BIC, M.AIC)
    
    def plotMCParams(self):
        if self.mc is None:
            raise Exception("MC not initialized. First call runMCMC()")
        plot(self.mc)

    def plotFit(self):
        if self.mc is None:
            raise Exception("MC not initialized. First call runMCMC()")
        plt.figure(figsize=(10,10))
        plt.title(self.name + " - " + self.data_name)
        plt.xlabel('Day')
        plt.ylabel('Deaths')
        plt.plot(self.mort_data, 'o', mec='black', color='black', label='Observed deaths')
        plt.plot(self.mortality_sim.stats()['mean'], color='red', linewidth=1, label='Model mean')
        y_min = self.mortality_sim.stats()['quantiles'][2.5]
        y_max = self.mortality_sim.stats()['quantiles'][97.5]
        plt.fill_between(range(0,self.t), y_min, y_max, color='r', alpha=0.3, label='Model 95% CI')
        plt.legend()
    
