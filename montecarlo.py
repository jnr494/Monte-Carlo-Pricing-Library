# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:32:27 2020

@author: Magnus Frandsen
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

class MonteCarlo:
    def __init__(self, model, option, n_sim):
        self.model = copy.deepcopy(model)
        self.option = copy.deepcopy(option)
        
        self.n_sim = int(n_sim)
        self.total_steps = option.get_req_steps() * option.get_maturity()
        
    def get_paths_vols(self, n_sim):
         temp_paths, temp_vols =  self.model.create_samples(n_sim = n_sim, 
                                                            n_steps = self.total_steps, 
                                                            time = self.option.get_maturity())
         return temp_paths, temp_vols
    
    def estimate_price(self, confidence = False):
        #Simulate payoffs
        self.payoffs = np.array([])
        
        n_runs = int(np.ceil(self.n_sim / (1e8 / self.model.n_random_var / self.total_steps)))
        #print(n_runs)
        
        for _ in range(n_runs):
            temp_paths, temp_vols = self.get_paths_vols(int(np.ceil(self.n_sim / n_runs)))
            temp_payoffs = np.exp(- self.model.rate * self.option.maturity) * self.option.get_payoff(temp_paths, temp_vols)
            self.payoffs = np.append(self.payoffs, temp_payoffs)
        
        #Calculate Price
        self.price = np.mean(self.payoffs)
        
        #Confidence interval
        if confidence is True:
            temp_conf = self.price + 1.96*np.array([-1,1])* np.std(self.payoffs) / np.sqrt(len(self.payoffs))
            
            return self.price, temp_conf
        else:
            return self.price
    
    def brute_force_confidence(self, reps):
        temp_prices = []
        
        for i in range(reps):
            temp_prices += [self.estimate_price(confidence = False)]
            print(i,temp_prices[-1])
            
        temp_prices = np.array(temp_prices)
        temp_mean_prices = np.mean(temp_prices)
        temp_std_prices = np.std(temp_prices)
        
        temp_conf = temp_mean_prices + 1.96*np.array([-1,1]) * temp_std_prices
        
        return temp_mean_prices, temp_conf, temp_std_prices, temp_prices
        
    def plot_mc_price(self):
        montes = np.cumsum(self.payoffs) / np.arange(1,len(self.payoffs)+1)
        plt.plot(montes)
        plt.plot(montes - 1.96 * np.std(self.payoffs) / np.sqrt(np.arange(1,len(self.payoffs)+1)), color = 'r',linewidth=0.5)
        plt.plot(montes + 1.96 * np.std(self.payoffs) / np.sqrt(np.arange(1,len(self.payoffs)+1)), color = 'r',linewidth=0.5)
        plt.ylim(np.mean(self.payoffs)*np.array([0.95,1.05]))
        plt.show()

