# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:04:59 2020

@author: Magnus Frandsen
"""

import models
import options
import montecarlo
import bmgenerator 
import time

bm_gen = bmgenerator.BMGenerator(method = 'sobolbb')
model = models.HestonModel(spot = 100, rate = 0.03, nu = 0.1, kappa = 3 , 
                           theta = 0.1, sigma = 0.3, rho = -0.5,bm_generator = bm_gen)

#model = models.BlackScholesModel(spot=100, rate=0, vol=0.2, bm_generator = bm_gen)
#model = models.SABRModel(spot = 100, rate = 0, beta = 0.2, sigma = 0.2, alpha = 0.4, rho = -0.8)
call_option = options.CallOption(strike = 80, maturity = 1)
#option = options.DoubleKnoutOut(option = call_option, maturity = 1, low_barrier = 70, high_barrier = 130)
#call_forward = options.ForwardCall(maturity = 1, steps = 100)

t0 = time.time()
mc_pricer = montecarlo.MonteCarlo(model= model, option = call_option, n_sim = 2**15)
print(mc_pricer.estimate_price(confidence = False))
t1 = time.time()
print('Time taken:', t1-t0)
mc_pricer.plot_mc_price()
#print(mc_pricer.brute_force_confidence(20))

