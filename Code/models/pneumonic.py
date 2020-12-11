from plague_model import PlagueModel
import pymc as pm
import numpy as np

class Pneumonic(PlagueModel, object):
    def __init__(self, mort_data, pop_size, data_name = "generic_data"):
        ps = dict()
        ps['beta'] = pm.Uniform('beta', 1e-9, 1.0)
        ps['gamma'] = 0.4
        ps['sus_frac'] = pm.Uniform('sus_frac', 1e-9, 1.)
        ps['S_0'] = pop_size * ps['sus_frac']
        ps['I_0'] = pm.Uniform('I_0', 1., 10.)
        ps['D_0'] = 1.
        vars = ps.values()
        mcVars = [ps['beta'], ps['sus_frac'], ps['I_0']]
        super(Pneumonic, self).__init__("Pneumonic", mort_data, pop_size, data_name=data_name, params=ps, inputVars=vars, mcmcVars=mcVars)


        def model(self):
            S_0 = self.params['S_0']
            I_0 = self.params['I_0']
            D_0 = self.params['D_0']
            t = self.t
            beta = self.params['beta']
            gamma = self.params['gamma']


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