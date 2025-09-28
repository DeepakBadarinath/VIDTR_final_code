#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 09:53:59 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU

#%%

class MarkovDecisionProcess:
    
    '''
    Class that represents a Markov Decision Process, with the states, actions,
    transitions and rewards
    '''
    
    def __init__(self, dimensions, state_spaces, action_spaces, time_horizon,
                 gamma, transition_kernels, reward_functions):
                 
        '''
        Parameters:                                                            
        -----------------------------------------------------------------------
        dimensions : list[int]
                     The dimensions of the state spaces of the MDP at the different timesteps
        
        state_spaces : list[DBU]
                       The state spaces of the MDP at the different timesteps                                   
         
        action_spaces : list[list]  
                        The action spaces for the MDP at the different timesteps                                      
                  
        time_horizon : int                                                     
                       The time horizon for the MDP                            
                                                                               
        gamma : float                                                          
                The discount factor for the MDP
        
        transition_kernels : list[function(s' \in state_spaces[t], s \in state_spaces[t], a \in action_spaces[t], state_space, action_space) \to [0,1]]
                             List of length T which consists of probability
                             transition maps.
                             Here the sum of the transition_kernels(s',s,a) for
                             all s' in states = 1
        
        reward_functions : list[function(state, action, state_space, action_space) \to \mathbb{R}]
                           List of length T which consists of reward_functions 
        
        '''
        
        self.dimensions = dimensions
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.time_horizon = time_horizon
        self.gamma = gamma                                                     
        self.transition_kernels = transition_kernels                           
        self.reward_functions = reward_functions     

    def maximum_over_actions(self, function, t):
        
        '''
        Given a function over states and actions, find the function only over
        states.
        
        Parameters:
        -----------------------------------------------------------------------
        function : function(s,a)
                   A function over states and actions for which we wish to get 
                   the map s \to max_A f(s,a)

        Returns:
        -----------------------------------------------------------------------
        max_function : function(s)
                       s \to \max_A f(s,a) is the function we wish to get
        
        '''
        def max_function(s):
            
            max_val = -np.inf
            
            for a in self.MDP.action_spaces[t]:
                if function(np.array(s),a) > max_val:
                    max_val = function(s,a)
            
            return max_val
                    
        return max_function
    
    
    def bellman_equation(self, t):
        
        '''
        Return the Bellman equation for the Markov Decision Process.           
        
        Assumes we know the true values from t+1 to T.                         
        
        Parameters:                                                                
        --------------------------------------------------------------------------
        t : float                                                               
            The time at which we wish to return the Bellman function for the MDP.
                                                                               
        Returns:                                                               
        -------------------------------------------------------------------------
        bellman_function : func                                                
                           The Bellman function of the MDP for the t'th timestep.

        '''
        def bellman_map(s,a):                                                   
            
            space = self.MDP.state_spaces[t]                                   
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            return self.MDP.reward_functions[t](np.array(s), a, space, action_space) + self.MDP.gamma * (
                    np.sum([self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space) * self.true_values[t+1](np.array(s_new)) 
                            for s_new in dbu_iterator]))                       
        
        return bellman_map
    
    
