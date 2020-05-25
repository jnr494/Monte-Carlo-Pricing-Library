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

#==================================================================
#Call option in Black Scholes without variance reduction

print('\nCall option in Black Scholes without variance reduction')

bm_gen_normal = bm_gen = bmgenerator.BMGenerator(method = 'normal') 
#method can be set to 'normal', 'a' antithetic, 'mm' (moment mathcing), 'sobol', sobolbb (Brownian Bridge) and 'sobolpca'

bs_model = models.BlackScholesModel(spot=100, rate=0, vol=0.2, bm_generator = bm_gen_normal) #Define model
call_option = options.CallOption(strike = 110, maturity = 1) #Define option
mc_pricer = montecarlo.MonteCarlo(model= bs_model, option = call_option, n_sim = 2**15) #Define MC-pricer

t0 = time.time()
price = mc_pricer.estimate_price(confidence = False) #Esimates price
t1 = time.time()
print('Estimated price:',price,'Time taken:',t1-t0) 

mc_pricer.plot_mc_price() #Plot convergence

#==================================================================
#Ordinary Call option in Black Scholes  with use of Sobol numbers and Brownian Bridge construction

print('\nCall option in Black Scholes with use of Sobol numbers and Brownian Bridge construction')

bm_gen_sobolbb = bm_gen = bmgenerator.BMGenerator(method = 'sobolbb') 

bs_model = models.BlackScholesModel(spot=100, rate=0, vol=0.2, bm_generator = bm_gen_sobolbb) #Define model
call_option = options.CallOption(strike = 110, maturity = 1) #Define option
mc_pricer = montecarlo.MonteCarlo(model= bs_model, option = call_option, n_sim = 2**15) #Define MC-pricer

t0 = time.time()
price = mc_pricer.estimate_price(confidence = False) #Esimates price
t1 = time.time()
print('Estimated price:',price,'Time taken:',t1-t0) 

mc_pricer.plot_mc_price() #Plot convergence

#==================================================================
#Call option in the Heston model without variance reduction

print('\nCall option in the Heston model without variance reduction')

bm_gen_normal = bm_gen = bmgenerator.BMGenerator(method = 'normal') 

heston_model = models.HestonModel(spot = 100, rate = 0.03, nu = 0.1, kappa = 3 , 
                                  theta = 0.1, sigma = 0.3, rho = -0.5,bm_generator = bm_gen) #Define model
call_option = options.CallOption(strike = 110, maturity = 1) #Define option
mc_pricer = montecarlo.MonteCarlo(model= heston_model, option = call_option, n_sim = 2**15) #Define MC-pricer

t0 = time.time()
price = mc_pricer.estimate_price(confidence = False) #Esimates price
t1 = time.time()
print('Estimated price:',price,'Time taken:',t1-t0) 

mc_pricer.plot_mc_price() #Plot convergence

#==================================================================
#Call option in the Heston model with use of Sobol numbers and Brownian Bridge construction

print('\nCall option in the Heston model with use of Sobol numbers and Brownian Bridge construction')

bm_gen_sobolbb = bm_gen = bmgenerator.BMGenerator(method = 'sobolbb') 

heston_model = models.HestonModel(spot = 100, rate = 0.03, nu = 0.1, kappa = 3 , 
                                  theta = 0.1, sigma = 0.3, rho = -0.5,bm_generator = bm_gen) #Define model

call_option = options.CallOption(strike = 110, maturity = 1) #Define option
mc_pricer = montecarlo.MonteCarlo(model= heston_model, option = call_option, n_sim = 2**15) #Define MC-pricer

t0 = time.time()
price = mc_pricer.estimate_price(confidence = False) #Esimates price
t1 = time.time()
print('Estimated price:',price,'Time taken:',t1-t0) 

mc_pricer.plot_mc_price() #Plot convergence

#==================================================================
#Double Knock-out-call-option in the Heston model with use of Sobol numbers and Brownian Bridge construction

print('\nDouble Knock-out-call-option in the Heston model with use of Sobol numbers and Brownian Bridge construction')

bm_gen_sobolbb = bm_gen = bmgenerator.BMGenerator(method = 'sobolbb') 

heston_model = models.HestonModel(spot = 100, rate = 0.03, nu = 0.1, kappa = 3 , 
                                  theta = 0.1, sigma = 0.3, rho = -0.5,bm_generator = bm_gen) #Define model

call_option = options.CallOption(strike = 110, maturity = None) #Define call option
dko_call_option = options.DoubleKnockOut(option = call_option, maturity = 1, low_barrier = 70, high_barrier = 130) #Define Knock-out option

mc_pricer = montecarlo.MonteCarlo(model= heston_model, option = dko_call_option, n_sim = 2**15) #Define MC-pricer

t0 = time.time()
price = mc_pricer.estimate_price(confidence = False) #Esimates price
t1 = time.time()
print('Estimated price:',price,'Time taken:',t1-t0) 

mc_pricer.plot_mc_price() #Plot convergence

print(mc_pricer.brute_force_confidence(20)) #Calculate confidence bands

