#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:00:10 2024

@author: badarinath

"""
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import math
import importlib
from itertools import product, combinations
import VIDTR_envs_pickle
from VIDTR_envs_pickle import GridEnv
import sys

#%%


sys.path.insert(0, r"C:\Users\innov\OneDrive\Desktop\IDTR-project\Code")

import IVIDTR_mV_fixes_memoiz_multithreaded as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc
import functools

#%%
                                                                                
importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(VIDTR_module)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs_pickle)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_mV_fixes_memoiz_multithreaded import VIDTR

from disjoint_box_union import DisjointBoxUnion as DBU

integrate_static = DBU.integrate_static
trajectory_integrate = DBU.trajectory_integrate
easy_integral = DBU.easy_integral


#%%

class VIDTR_grid:
    
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, goal, time_horizon, gamma, eta, rho,
                 max_conditions = np.inf, reward_coeff=1.0, friction = 1.0):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     Dimension of the grid 
        
        center : np.array
                 Center of the grid
                 
        side_lengths : np.array
                       Side lengths of the grid
                       
        stepsizes : np.array
                    Stepsizes for the grid
        
        max_lengths : np.array 
                      Maximum lengths for the grid
        
        max_complexity : int
                         Maximum complexity for the tree 
        
        goal : np.array
               Location of the goal for the 2D grid problem
               
        time_horizon : int
                       Time horizon for the VIDTR problem
                       
        gamma : float
                Discount factor
        
        eta : float
              Splitting promotion constant    
        
        rho : float
              Condition promotion constant
        
        max_conditions : int
                         The maximum conditions we iterate over for the VIDTR
        
        reward_coeff : float
                       The scaling for the reward
        
        friction : float
                   The friction constant; higher values here implies greater penalization
                   for the VIDTR algorithm

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
        VIDTR_MDP : markov_decision_processes
                    The Markov Decision Process represented in the algorithm
        
        algo : VIDTR_algo
               The algorithm representing VIDTR
        '''
        self.dimensions = dimensions
        self.center = center
        self.side_lengths = side_lengths
        self.stepsizes = stepsizes
        self.max_lengths = max_lengths
        self.max_complexity = max_complexity
        self.goal = goal
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.friction = friction
        self.reward_coeff = reward_coeff
        self.max_conditions = max_conditions        
        
        self.env = GridEnv(dimensions, center, side_lengths, goal,
                           stepsizes = stepsizes, reward_coeff=reward_coeff,
                           friction = friction)
        
        self.transitions = [self.env.transition for t in range(time_horizon)]
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes, max_conditions = max_conditions)
        
    
    def generate_random_trajectories(self, N, random_seed = 42):
        '''
        Generate N trajectories from the VIDTR grid setup where we take a
        random action at each timestep and we choose a random initial state
        
        Returns:
        -----------------------------------------------------------------------
           obs_states : list[list]
                        N trajectories of the states observed
        
           obs_actions : list[list]
                         N trajectories of the actions observed
           
           obs_rewards : list[list]
                         N trajectories of rewards obtained                    
           
        '''
        
        obs_states = []
        obs_actions = []
        obs_rewards = []
        
        for traj_no in range(N):
            obs_states.append([])
            obs_actions.append([])
            obs_rewards.append([])
            s = np.squeeze(self.VIDTR_MDP.state_spaces[0].pick_random_point(random_seed = random_seed))  
            obs_states[-1].append(s)
            
            for t in range(self.time_horizon):
                
                a = random.sample(self.actions[t], 1)[0]
                r = self.rewards[t](s,a)
                
                s = self.env.move(s,a)
                obs_states[-1].append(s)
                obs_actions[-1].append(a)
                obs_rewards[-1].append(r)
                
            
        return obs_states, obs_actions, obs_rewards
            
#%%
'''
Tests GridEnv
'''

if __name__ == '__main__':
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([0, 0])
    time_horizon = 5
    gamma = 0.9
    max_lengths = [4 for t in range(time_horizon)]
    stepsizes = 1.0
    max_complexity = 2
    etas = [-0.05 * 9 ,-0.05 * 8,-0.05 * 7,-0.05*6 ,-0.05 * 5]
    rhos = 0.1
    reward_coeff = 1.5 * 3
    friction = 2.0
    
    max_conditions = np.inf
    
    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                            stepsizes, max_lengths, max_complexity, goal,
                            time_horizon, gamma, etas, rhos,
                            max_conditions = max_conditions, reward_coeff = reward_coeff,
                            friction = friction)
    
    #%%
    
    '''
    Tests for optimal_policies and values
    '''
    
    optimal_policies, optimal_value_funcs = grid_class.algo.compute_optimal_policies()
    s_vals = [np.array([0,0]), np.array([0,1]), np.array([-2,0])]
   
    
#%%%%
   
    '''
    VIDTR - optimal policy plots                                           
    '''
   
    for t in range(grid_class.time_horizon):                                   
      
      optimal_policy = optimal_policies[t]
          
      grid_class.env.plot_policy_2D(optimal_policy, title=f'Optimal policy at time {t}',
                                    saved_fig_name = f'Optimal_policy_{t}')

    #%%%
    
    t = 0
    
    int_bellman_map = functools.partial(VIDTR.fixed_reward_function,
                                        t = t,
                                        MDP = grid_class.VIDTR_MDP)
    
    cond_DBU = DBU(1, 2, np.array([[1,6]]), np.array([[-2.5,0]]))
    
    
    for a in grid_class.algo.MDP.action_spaces[t]:
    
        fixed_bellman_function = functools.partial(int_bellman_map,
                                                   a = a)
        
        new_space = grid_class.algo.MDP.state_spaces[t]
        dbu_iter_class = disjoint_box_union.DBUIterator(new_space)
        dbu_iterator = iter(dbu_iter_class)
        
        print(f'At action {a}, the integral of the FB map is')
        print(integrate_static(cond_DBU, fixed_bellman_function))
        
        
   #%%%
   
    r = 1000
    N = 5000
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N, random_seed=r)
   
    error_file_name = 'vidtr_errors_multithread.csv'
   
    optimal_DBUs, optimal_actions = grid_class.algo.compute_interpretable_policies(integration_method='static_integral',
                                                                                   conditions_string = 'all',
                                                                                   obs_states = obs_states,
                                                                                   store_errors=True,
                                                                                   error_file_name = error_file_name) 
                                                             
   #%%%                                                                         
    '''                                                                        
    VIDTR - plot errors                                                        
    '''                                                                        
    grid_class.algo.plot_errors()                                              
                                                                               

  #%%
    '''
    VIDTR - get interpretable policy                                           
    '''
    
    for t in range(grid_class.time_horizon):                                   
       
       int_policy = VIDTR.get_interpretable_policy_dbus(grid_class.algo.stored_DBUs[t],
                                                        grid_class.algo.optimal_actions[t])
           
       grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}',
                                     saved_fig_name = f'VIDTR_multithreaded_{t}')

   
   #%%%