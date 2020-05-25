# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:58:39 2020

@author: Magnus Frandsen
"""

import numpy as np
from abc import ABCMeta, abstractmethod

class Option(metaclass=ABCMeta):
    def __init__(self,maturity, req_steps):
        self.maturity = maturity
        self.req_steps = req_steps
        
    def get_req_steps(self):
        return self.req_steps
    
    def get_maturity(self):
        return self.maturity
    
    @abstractmethod
    def get_payoff(self,sample_paths,sample_vols):
        pass
    
    
class CallOption(Option):
    def __init__(self,strike,maturity):
        self.strike = strike
        
        super().__init__(maturity = maturity, req_steps = 1)
        
    def get_payoff(self,sample_paths,sample_vols):
        payoffs = np.maximum(0,sample_paths[:,-1] - self.strike)
        return payoffs.flatten()
    
class PutOption(Option):
    def __init__(self,strike,maturity):
        self.strike = strike
        
        super().__init__(maturity = maturity, req_steps = 1)
        
    def get_payoff(self,sample_paths,sample_vols):
        payoffs = np.maximum(0,self.strike - sample_paths[:,-1])
        return payoffs.flatten()

class Digital(Option):
    def __init__(self,strike,maturity):
        self.strike = strike
        
        super().__init__(maturity = maturity, req_steps = 1)
    
    def get_payoff(self,sample_paths,sample_vols):
        payoffs = (sample_paths[:,-1] >= self.strike) * 1
        return payoffs.flatten()

class ForwardCall(Option):
    def __init__(self,maturity,steps):
        super().__init__(maturity = maturity, req_steps = steps)
    
    def get_payoff(self,sample_paths,sample_vols):
        payoffs = sample_paths[:,-1] - np.min(sample_paths, axis = 1)
        return payoffs.flatten()

class ForwardPut(Option):
    def __init__(self,maturity,steps):
        super().__init__(maturity = maturity, req_steps = steps)
    
    def get_payoff(self,sample_paths,sample_vols):
        payoffs = - np.max(sample_paths, axis = 1) - sample_paths[:,-1] 
        return payoffs.flatten()


class BarrierOption(Option,metaclass = ABCMeta):
    def __init__(self,option, maturity, req_steps):
        self.option = option
        super().__init__(maturity, req_steps)
    
    def get_barrier_p(self, sample_paths, sample_vols, barrier):
        n_steps = len(sample_paths[0,:]) - 1
        dt = self.maturity / n_steps
        
        p_high = np.array([np.float(1)]*len(sample_paths[:,0]))
        for i in range(n_steps):
            numerator = - 2 * np.log(barrier / sample_paths[:,i]) \
                        * np.log(barrier / sample_paths[:,i+1])
            denominator = sample_vols[:,i]**2 * dt
            
            p_high *= (1 - np.exp(numerator / denominator))
        
        return p_high.flatten()
    
    @abstractmethod
    def get_payoff(self,sample_paths,sample_vols):
        pass

class UpAndOut(BarrierOption):
    def __init__(self,option, maturity, high_barrier):
        self.high_barrier = high_barrier
        
        super().__init__(option = option, maturity = maturity, req_steps = 200)
    
    def get_payoff(self, sample_paths, sample_vols, brownian_bridge = True):
        pre_payoffs = self.option.get_payoff(sample_paths,sample_vols).flatten()
        
        barrier_hit = np.prod((sample_paths <= self.high_barrier)*1, axis =1)
        payoffs = pre_payoffs * barrier_hit
        
        payoffs = payoffs * self.get_barrier_p(sample_paths,sample_vols,self.high_barrier) if brownian_bridge else payoffs          
        return payoffs

class DownAndOut(BarrierOption):
    def __init__(self,option, maturity, low_barrier):
        self.low_barrier = low_barrier
        
        super().__init__(option = option, maturity = maturity, req_steps = 200)
        
    
    def get_payoff(self, sample_paths, sample_vols, brownian_bridge = True):
        pre_payoffs = self.option.get_payoff(sample_paths,sample_vols).flatten()
        
        barrier_hit = np.prod((sample_paths >= self.low_barrier)*1, axis =1)
        payoffs = pre_payoffs * barrier_hit
        
        payoffs = payoffs * self.get_barrier_p(sample_paths,sample_vols,self.low_barrier) if brownian_bridge else payoffs          
        return payoffs

class DoubleKnockOut(BarrierOption):
    def __init__(self,option, maturity, low_barrier, high_barrier):
        self.low_barrier = low_barrier
        self.high_barrier = high_barrier
        
        super().__init__(option = option, maturity = maturity, req_steps = 200)

    def get_payoff(self, sample_paths, sample_vols, brownian_bridge = True):
        pre_payoffs = self.option.get_payoff(sample_paths,sample_vols).flatten()
        
        low_barrier_hit = np.prod((sample_paths >= self.low_barrier)*1, axis =1)
        high_barrier_hit = np.prod((sample_paths <= self.high_barrier)*1, axis =1)
        
        payoffs = pre_payoffs * low_barrier_hit * high_barrier_hit
        
        if brownian_bridge is True:
            p_adjustment = self.get_barrier_p(sample_paths,sample_vols,self.low_barrier) \
                         * self.get_barrier_p(sample_paths,sample_vols,self.high_barrier)
                         
            payoffs = payoffs * p_adjustment
            
        return payoffs
