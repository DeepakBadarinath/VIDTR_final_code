#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:36:08 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import inspect
import math

import markov_decision_processes as mdp_module
import disjoint_box_union                                                                          
import constraint_conditions as cc                                              
import pandas as pd
import functools
import random


import itertools
import sys
sys.setrecursionlimit(9000)  # Set a higher limit if needed
                                                                                           
#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)

from disjoint_box_union import DisjointBoxUnion as DBU

integrate_static = DBU.integrate_static
trajectory_integrate = DBU.trajectory_integrate
easy_integral = DBU.easy_integral

from concurrent.futures import ProcessPoolExecutor, as_completed

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
#torch.manual_seed(SEED)

#%%

def zero_function(s):
    return 0.0


def constant_map(s, c):
    return c


#%%%


class MaxOverActions:
    """
    A picklable callable equivalent of the nested max_function.
    """
    def __init__(self, t, function, MDP):
        self.t = t
        self.function = function
        self.MDP = MDP

    def __call__(self, s):
        max_val = -np.inf
        for a in self.MDP.action_spaces[self.t]:
            val = self.function(np.array(s), a)
            if val > max_val:
                max_val = val
        return max_val


def maximum_over_actions(t, function, MDP):
    """
    Given a function f(s, a), returns a callable that maps s → max_a f(s, a).
    This version avoids nested functions and is picklable.
    """
    return MaxOverActions(t, function, MDP)

#%%%
'''
Multithreading top-level helper functions
'''

class BellmanMap:
    """Picklable equivalent of the nested bellman_map(s, a)."""
    def __init__(self, t, MDP, optimal_value_funcs):
        self.t = t
        self.MDP = MDP
        self.optimal_value_funcs = optimal_value_funcs

    def __call__(self, s, a):
        MDP, t = self.MDP, self.t

        curr_space = MDP.state_spaces[t]

        if len(MDP.state_spaces) <= (t + 1):
            new_space = curr_space
        else:
            new_space = MDP.state_spaces[t + 1]

        dbu_iter_class = disjoint_box_union.DBUIterator(new_space)
        dbu_iterator = iter(dbu_iter_class)

        # Base case: final time step
        if t == (MDP.time_horizon - 1):
            return MDP.reward_functions[t](np.array(s), a)

        # Recursive case: Bellman equation
        r = MDP.reward_functions[t](np.array(s), a)
        T_times_V = 0.0

        for s_new in dbu_iterator:
            kernel_eval = MDP.transition_kernels[t](np.array(s_new), np.array(s), a)
            vals_element = self.optimal_value_funcs[t + 1](np.array(s_new))
            T_times_V += kernel_eval * vals_element

        return r + MDP.gamma * T_times_V


def bellman_equation(t, MDP, optimal_value_funcs):
    """
    Return a picklable Bellman function for timestep t of the MDP.
    """
    return BellmanMap(t, MDP, optimal_value_funcs)

#%%%%%
'''
Conversion of dict to function

'''

class PicklableFunction:
    """A lightweight callable class to wrap dictionary lookups."""
    __slots__ = ('f_dict', 'default_value')

    def __init__(self, f_dict, default_value=0):
        self.f_dict = f_dict
        self.default_value = default_value

    def __call__(self, s):
        return self.f_dict.get(tuple(s), self.default_value)

    def __repr__(self):
        return f"<PicklableFunction with {len(self.f_dict)} entries>"

# -----------------------------------------------------------------------
# 2. This factory function just instantiates the class
# -----------------------------------------------------------------------
def convert_dict_to_function(f_dict, S, default_value=0):
    """
    Given a dictionary f_dict, redefine it such that we get a picklable
    callable f mapping states in S to actions (or values).
    """
    return PicklableFunction(f_dict, default_value)



#%%%%
'''

Multithreading top-level worker functions

'''

def compute_error_for_cond_action(args):

    (i, cond_DBU, act_no, a, t, l, rho, obs_states,
     function_dicts, integration_method, store_errors, conditions_string) = args

    maxim_bellman_function = function_dicts['maxim_bellman_function']
    fixed_bellman_function = function_dicts['fixed_bellman_function']
    eta_function = function_dicts['eta_function']
    
    def total_bellman_function(s):
        
        return maxim_bellman_function(s = s) - fixed_bellman_function(s = s)

    # choose integration method
    if integration_method == 'trajectory_integrate':
        bellman_error = trajectory_integrate(obs_states, total_bellman_function, cond_DBU, t)
        const_error = trajectory_integrate(obs_states, eta_function, cond_DBU, t)
    else:
        bellman_error = integrate_static(
            cond_DBU,
            total_bellman_function,
            check_positivity=True,
            function_name=f'Bellman_function_time_{t}_cond_no_{i}_action_{act_no}'
        )
        const_error = integrate_static(cond_DBU, eta_function)

    complexity_error = rho * cond_DBU.complexity * cond_DBU.no_of_points()
    total_error = bellman_error + const_error + complexity_error

    result_row = None
    if store_errors:
        result_row = {
            'Time': t,
            'Length': l,
            'Centre': cond_DBU.centres,
            'Lengths': cond_DBU.lengths,
            'Bellman_error': bellman_error,
            'Constant_error': const_error,
            'Action': a,
            'Complexity_error': complexity_error,
            'Total_error': total_error,
            'Integration_method': integration_method,
            'Conditions_string': conditions_string
        }

    return (total_error, const_error, complexity_error, bellman_error)

#%%%%


def bellman_equation_I(MDP, t, int_value_functions):
    
    '''
    Return the interpretable Bellman equation for the Markov Decision Process.
    
    Assumes we know the interpretable value function from timestep t+1 to T.
    
    Parameters:
    ---------------------------------------------------------------------------
    t : float
        The time at which we wish to return the interpretable Bellman equation
    
    Returns:
    ---------------------------------------------------------------------------
    int_bellman_function : func
                           The Interpretable Bellman function for the MDP for the t'th timestep
    
    '''
            
    #ic(f'We ask to evaluate the bellman map at timestep {t}')
    if t == MDP.time_horizon - 1:
        return functools.partial(VIDTR.fixed_reward_function,
                                 t = t,
                                 MDP = MDP)
    
    else:
        
        int_bellman_function = functools.partial(VIDTR.bellman_value_function_I,
                                                 t=t,
                                                 MDP=MDP,
                                                 int_value_function_next = int_value_functions[t+1]
                                                 )
        
        return int_bellman_function

#%%%

class VIDTR:
    
    def __init__(self, MDP, max_lengths,
                 etas, rhos, max_complexity,
                 stepsizes, max_conditions = math.inf):                        
        '''
        Value-based interpretable dynamic treatment regimes; Generate a tree based
        policy while solving a regularized interpretable form of the Bellmann 
        equation. In this module we assume that the state spaces are time dependent.
        
        Parameters:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
              The underlying MDP from where we want to get interpretable policies
              
        max_lengths : list[T] or int
                      The max depth of the tree upto the T timesteps
        
        etas : list[T] or int
               Volume promotion constants
               Higher this value, greater promotion in the splitting process    
                                                                               
        rhos : list[T] or int                                                            
               Complexity promotion constants                                    
               Higher this value, greater the penalization effect of the complexity 
               splitting process                                                
                                                                               
        max_complexity : int or list                                                   
                         The maximum complexity of the conditions; maximum number of 
                         and conditions present in any condition               
                                                                               
        stepsizes : list[np.array((1, MDP.states.dimension[t])) for t in range(time_horizon)] or float or int        
                    The stepsizes when we have to integrate over the DBU       
        
        max_conditions : int or list                                           
                         The maximum number of conditions per time and lengthstep
                         If None then all the conditions will be looked at     
        '''
        
        self.MDP = MDP
        self.time_horizon = self.MDP.time_horizon
        
        if type(max_lengths) == float or type(max_lengths) == int:
            max_lengths = [max_lengths for t in range(self.MDP.time_horizon)]
        
        self.max_lengths = max_lengths
        
        if type(etas) == float or type(etas) == int:
            etas = [etas for t in range(self.MDP.time_horizon)]
        
        self.etas = etas
        
        if type(rhos) == float or type(rhos) == int:
            rhos = [rhos for t in range(self.MDP.time_horizon)]

        self.rhos = rhos
        
        if type(stepsizes) == float or type(stepsizes) == int:
            stepsizes = [np.ones((1, MDP.state_spaces[t].dimension)) for t in range(self.time_horizon)]
        
        self.stepsizes = stepsizes
        
        if type(max_complexity) == int:
            max_complexity = [max_complexity for t in range(self.MDP.time_horizon)]
        
        self.max_complexity = max_complexity
        
        self.true_values = [zero_function for t in range(self.MDP.time_horizon+1)]
        
        if ((type(max_conditions) == int) or (max_conditions == math.inf)):
            max_conditions = [max_conditions for t in range(self.MDP.time_horizon)]
        
        self.max_conditions = max_conditions
        #ic(f'The max conditions is {self.max_conditions}')
        
    @staticmethod
    def redefine_function(f, s, a):                                            
        '''
        Given a function f, redefine it such that f(s) is now a                
                                                                                
        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            Old function we wish to redefine                                   
        s : type(domain(function))                                             
            The point at which we wish to redefine f                           
        a : type(range(function))                                                
            The value taken by f at s                                          
 
        Returns:                                                                  
        -----------------------------------------------------------------------
        g : function                                                           
            Redefined function                                                 
                                                                                
        '''
        def g(state):
            if np.sum((np.array(state)-np.array(s))**2) == 0:
                return a
            else:
                return f(state)
        return g
        
    @staticmethod
    def convert_function_to_dict_s_a(f, S):
        '''
        Given a function f : S \times A \to \mathbb{R}                         
        Redefine it such that f is now represented by a dictonary              

        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            The function that is to be redefined to give a dictonary           
            
        S : iterable version of the state space                                
            iter(DisjointBoxUnionIterator)                                     

        Returns:
        -----------------------------------------------------------------------
        f_dict : dictionary
                The function which is now redefined to be a dictonary

        '''
        f_dict = {}
        for s in S:
            f_dict[tuple(s)] = f(s)
        
        return f_dict
    
    
    def memoizer(self, t, f):
        
        '''
        Given a function f over the MDP state space at time t, create it's memoized version
        We do this by creating f_dict where we have f_dict[tuple(s)] = f(s)
        
        '''
        
        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
        
        state_iterator = iter(dbu_iter_class)
        
        f_dict = {}
        
        for s in state_iterator:
            
            f_dict[tuple(s)] = f(s)
        
        f_memoized = convert_dict_to_function(f_dict, state_iterator)
        return f_memoized
    
    
    def compute_optimal_policies(self):
        
        '''
        Compute the true value functions at the different timesteps.
        
        Note that the way we do this is by employing the dict based storing method to avoid 
        recursive calls. This works by us computing the value function for all state points and timestep t+1
        
        Stores:
        -----------------------------------------------------------------------
        optimal_values : list[function]
                         A list of length self.MDP.time_horizon which represents the 
                         true value functions at the different timesteps.
        
        optimal_policies : list[function]
                           The list of optimal policies for the different timesteps 
                           for the MDP.
        '''
        #zero_value = lambda s : 0
        zero_value_dicts = []
        const_action_dicts = []
        
        zero_func = functools.partial(constant_map,
                                      c = 0)
        
        self.optimal_policy_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        self.optimal_value_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        
        for t in range(self.time_horizon):
            
            #Setting up this iter_class is taking a lot of time-why?
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            zero_dict = {}
            const_action_dict = {}
            
            for s in state_iterator:
                
                zero_dict[tuple(s)] = 0
                const_action_dict[tuple(s)] = self.MDP.action_spaces[t]
                
            with open(f"We finish {t}-level zero_dict computation", "w") as f:
                f.write("Zero-level dict computation is now done")
            
            zero_value_dicts.append(zero_dict)
            const_action_dicts.append(const_action_dict)
        
        
        optimal_policy_dicts = const_action_dicts
        
        optimal_value_dicts = zero_value_dicts
        
                                                                         
        for t in np.arange(self.time_horizon-1, -1, -1):     
                          
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            
            state_iterator = iter(dbu_iter_class)
            
            for s in state_iterator:
                
                max_val = -np.inf
                
                
                for a in self.MDP.action_spaces[t]:
                                                    
                    bellman_value = bellman_equation(t, self.MDP, self.optimal_value_funcs)(s,a)
                    
                    if bellman_value > max_val:                                 
                        
                        max_val = bellman_value
                        best_action = a
                                                                                
                        optimal_policy_dicts[t][tuple(s)] = best_action                       
                        optimal_value_dicts[t][tuple(s)] = max_val                        
            
            
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_func = convert_dict_to_function(optimal_policy_dicts[t],
                                                                 state_iterator)
            
            dbu_iter_class_1 = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator_new = iter(dbu_iter_class_1)
            
            optimal_value_func = convert_dict_to_function(optimal_value_dicts[t],
                                                          state_iterator_new)
            
            with open(f"We finish {t}-optimal-policy and value function computation", "w") as f:
                f.write(f"{t}-level optimal policy and value function computation is now done")
            
            self.optimal_policy_funcs[t] = optimal_policy_func
            self.optimal_value_funcs[t] = optimal_value_func
            
        optimal_policy_funcs = []
        optimal_value_funcs = []
        
        for t in range(self.MDP.time_horizon):
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_funcs.append(convert_dict_to_function(optimal_policy_dicts[t],
                                                                 state_iterator,
                                                                 default_value=self.MDP.action_spaces[t][0]))
            
            optimal_value_funcs.append(convert_dict_to_function(optimal_value_dicts[t],
                                                                state_iterator,
                                                                default_value=0.0))
            
            with open(f"We finish {t}-optimal-policy and value function appending", "w") as f:
                f.write(f"{t}-level optimal policy and value function appending is now done")
            
        self.optimal_policy_funcs = optimal_policy_funcs
        self.optimal_value_funcs = optimal_value_funcs
        
        
        return optimal_policy_funcs, optimal_value_funcs
    
    def constant_eta_function(self, t):                                         
        '''
        Return the constant \eta function for time t

        Parameters:
        -----------------------------------------------------------------------
        t : int                                                                
            Time step                                                           

        Returns:
        -----------------------------------------------------------------------
        f : function
            Constant eta function at time t                                    
                                                                                 
        '''                                                                                               
        f = lambda s,a: self.etas[t]
        return f
    
    @staticmethod
    def fixed_reward_function(t, s, a, MDP, debug = False):
        
        '''
        For the MDP as in the function parameter, return the reward function.
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            Timestep at which we return the reward function
        
        s : state_point
            The point on the state space at which we return the reward function
        
        a : action
            The action we take at this reward function
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the fixed reward function  
        
        '''
        
        return MDP.reward_functions[t](s, a)
    
    def bellman_equation(self, t):
        '''
        Return the Bellman equation for the Markov Decision Process.           
        
        Assumes we know the true values from t+1 to T.                         
        
        Parameters:                                                                
        -----------------------------------------------------------------------
        t : float                                                               
            The time at which we wish to return the Bellman function for the MDP.
                                                                               
        Returns:                                                               
        -----------------------------------------------------------------------
        bellman_function : func                                                
                           The Bellman function of the MDP for the t'th timestep.

        '''
        def bellman_map(s,a):                                                   
            
            curr_space = self.MDP.state_spaces[t]
            
            if len(self.MDP.state_spaces) <= (t+1):
                new_space = curr_space
            else:
                new_space = self.MDP.state_spaces[t+1]
                                   
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(new_space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            if (t == (self.MDP.time_horizon - 1)):
            
                return self.MDP.reward_functions[t](np.array(s), a)
            
            else:
                
                r = self.MDP.reward_functions[t](np.array(s), a)
                T_times_V = 0
                for s_new in dbu_iterator:
                
                    kernel_eval = self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a)
                    vals_element = self.optimal_value_funcs[t+1](np.array(s_new))
                    
                    adding_element = kernel_eval * vals_element
                    T_times_V += adding_element
                
                
                return r + self.MDP.gamma * T_times_V             
        
        return bellman_map                                                     
    
    
    @staticmethod
    def bellman_value_function_I(t, s, a, MDP,
                                 int_value_function_next, debug = False):
        
        '''
        For the MDP, return the interpretable bellman_value_function at timestep t.
        
        This is given by:
            r_t(s,a) + \gamma \sum_{s' \in S} P^a_t(s',s,a') V^I_{t+1}(s')
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            The timestep at which we compute the Bellman value function.
        
        s : state_point
            The point on the state space at which we return the Bellman value function.
        
        a : action
            The action we take on the Bellman value function.
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the Bellman value function.
        
        '''
        
        space = MDP.state_spaces[t]                                   
        action_space = MDP.action_spaces[t]                           
        
        if len(MDP.state_spaces) <= t+1:
            iter_space = MDP.state_spaces[t]
        else:
            iter_space = MDP.state_spaces[t+1]
                                                                                
        dbu_iter_class = disjoint_box_union.DBUIterator(iter_space)              
        dbu_iterator = iter(dbu_iter_class)                                     
        
        if debug:
         
            total_sum = MDP.reward_functions[t](np.array(s), a, space, action_space)
            for s_new in dbu_iterator:

                total_sum += int_value_function_next(np.array(s_new)) * MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space)
            
        return MDP.reward_functions[t](np.array(s), a) + MDP.gamma * (
                np.sum([[MDP.transition_kernels[t](np.array(s_new), np.array(s), a) * int_value_function_next(np.array(s_new)) 
                        for s_new in dbu_iterator]]))
        
    
    @staticmethod
    def last_step_int_value_function(t, int_policy, MDP, debug = False):
        
        def last_step_val_function(s):
            
            return VIDTR.fixed_reward_function(t, s, int_policy(s),
                                               MDP, debug=debug)
        
        return last_step_val_function
    
    
    @staticmethod
    def general_int_value_function(t, int_policy, MDP,
                                   next_int_val_function, debug = False):
        
        def interpretable_value_function(s):
            return VIDTR.bellman_value_function_I(t, s, int_policy(s),
                                                  MDP,
                                                  next_int_val_function,
                                                  debug=debug)
        
        return interpretable_value_function


    def int_value_checks(self, t, int_value_function, int_policy):
        
        '''
        Check whether the int_value_function for time t and for int_policy is 
        greater than the optimal_value_function for the same timestep.
        '''
        
        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
        state_iterator = iter(dbu_iter_class)
        
        for s in state_iterator:
            
            int_value = int_value_function(s)
            optimal_value = self.optimal_value_funcs[t](s)
            
            if int_value > optimal_value:
                
                optimal_value = self.optimal_value_funcs[t](s)
                raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
    

    def compute_errors_parallel(self, condition_DBUs, t, l, DBU, obs_states,
                                integration_method, store_errors, conditions_string, max_workers=None):
        '''
        Compute the evaluation of the objective for all the different cond_DBUs
        and the actions in parallel
        
        Store errors functionality is not working
        
        -----------------------------------------------------------------------
        '''
        print(f'We are at timestep {t}, lengthstep {l}')
        
        tasks = []
        function_dicts = {}
        bellman_map = bellman_equation(t, self.MDP, self.optimal_value_funcs)
        function_dicts['maxim_bellman_function'] = maximum_over_actions(t, bellman_map, self.MDP)
        
        function_dicts['eta_function'] = functools.partial(constant_map, c = self.etas[t])
        
        
        for i, cond_DBU in enumerate(condition_DBUs):
            for act_no, a in enumerate(self.MDP.action_spaces[t]):
                
                int_bellman_map = bellman_equation_I(self.MDP, t, self.int_value_functions)
                
                function_dicts['fixed_bellman_function'] = functools.partial(int_bellman_map,
                                                                             a = a)
                
                tasks.append((i, cond_DBU, act_no, a, t, l, self.rhos[t], obs_states,
                              function_dicts, integration_method, store_errors, conditions_string))
    
        results = []
        error_df = pd.DataFrame()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_error_for_cond_action, args) for args in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
        
        # Find minimum total error
        min_index, min_error_tuple = min(enumerate(results), key=lambda x: x[1][0])
        (min_total_error, min_const_error, min_complexity_error, min_bellman_error) = min_error_tuple
        
        # Retrieve the corresponding cond_DBU and action
        (_, optimal_cond_DBU, _, optimal_action, *_) = tasks[min_index]
        
        print(f"✅ Minimum total error = {min_total_error:.6f} at index {min_index}")
        print(f"Action: {optimal_action}")
        print(f"Condition centre: {optimal_cond_DBU.centres}")
        
        if store_errors:
            valid_rows = [r[-1] for r in results if len(r) == 5 and r[-1] is not None]
            error_df = pd.DataFrame(valid_rows)
        
        return (optimal_cond_DBU, optimal_action,
                min_total_error, min_const_error,
                min_complexity_error, min_bellman_error, error_df)
        
    
    def compute_interpretable_policies(self,
                                       integration_method = 'trajectory_integrate',
                                       integral_percent = 0.5, debug = False,
                                       obs_states=None, conditions_string = 'all',
                                       store_errors = False,
                                       error_file_name = 'error_df.csv'):                                                               
        
        '''                                                                    
        Compute the interpretable policies for the different length and        
        timesteps given a DBUMarkovDecisionProcess.                             
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        integration_method : string
                             The method of integration we wish to use -
                             trajectory integratal versus DBU based integral
        
        integral_percent : float
                           The percent of points we wish to sample from 
        
        debug : bool
                Add additional ic statements to debug the plots
        
        obs_states : list[list]
                     The observed states at the different time and lengthsteps
        
        conditions_string : string
                            'all' or 'order_stats' -> Use all possible conditions by going over
                            each dimension and state_differences versus conditions given by those over
                            the order statistics
        
        Stores:
        -----------------------------------------------------------------------
        optimal_conditions :  list[list]
                              condition_space[t][l] gives the optimal condition at
                              timestep t and lengthstep l
        
        optimal_errors : list[list]
                         errors[t][l] represents the error obtained at timestep t and 
                         length step l
        
        optimal_actions : list[list]
                          optimal_intepretable_policies[t][l] denotes
                          the optimal interpretable policy obtained at 
                          time t and length l
        
        stored_DBUs :   list[list]
                        stored_DBUs[t][l] is the DBU stored at timestep t and  
                        lengthstep l for the final policy
        
    optimal_fixed_bellman_errors : list[list]    
                                   optimal_fixed_bellman_errors[t][l] is the optimal fixed Bellman 
                                   error for timestep t and lengthstep l
    
    optimal_maxim_bellman_errors : list[list]
                                   optimal_maxim_bellman_errors[t][l] is the optimal maxim Bellman
                                   error for timestep t and lengthstep l
    
        
        stepsizes :  np.array(self.DBU.dimension) or int or float
                     The length of the stepsizes in the different dimensions
        
        total_error : float
                      The total error in the splitting procedure
        
        total_bellman_error : float
                              The total Bellman error resulting out of the splitting 
                    

        int_policies : list[list[function]]
                       optimal_intepretable_policies[t][l] denotes
                       the optimal interpretable policy obtained at 
                       time t and length l
        
        store_errors : bool
                       Store the errors for the different metrics for different times and lengths
        
        
        error_file_name : string
                          Store the name of the error file
        
        Returns:
        -----------------------------------------------------------------------
        optimal_conditions, optimal_actions : Optimal condition spaces and optimals
                                              for the given time and lengthstep
                                              respectively
        
        '''
        
        stored_DBUs = []
        optimal_actions = []
        optimal_bellman_errors = []
        int_policies = []
        zero_func =  functools.partial(constant_map, c = 0)
        int_value_functions = [zero_func for t in range(self.MDP.time_horizon)]
        
        self.int_value_functions = int_value_functions
        
        optimal_errors = []
        optimal_fixed_bellman_errors = []
        optimal_maxim_bellman_errors = []
        
        # Start from the back and go backwards each step

        # Initialize empty dataframe
        if store_errors:
            col_names = ['Time', 'Length', 'Centre', 'Lengths',
                         'Bellman_error', 'Constant_error', 'Complexity_error', 'Total_error',
                         'Integration_method', 'Conditions_string']
            
            error_df = pd.DataFrame(columns=col_names)
        
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            
            all_conditions = []
            all_condition_DBUs = []
            int_policies = [[] ,*int_policies]
            
            if conditions_string == 'all':
                all_conditions = self.MDP.state_spaces[t].generate_all_conditions(self.max_complexity[t])
            
            elif conditions_string == 'order_stats':
                
                obs_states_at_t = [obs_states[i][t] for i in range(len(obs_states))]
                all_conditions = self.MDP.state_spaces[t].generate_conditions_from_observations(obs_states_at_t, self.max_complexity[t])
               
            state_bounds = self.MDP.state_spaces[t].get_total_bounds()
            remaining_space = self.MDP.state_spaces[t]
            
            total_error = 0
            total_bellman_error = 0
            
            all_condition_dicts = {}
            
            access_last_lengthstep = True
            
            for i,c in enumerate(all_conditions):
                if c != None:
                    
                    con_DBU = DBU.condition_to_DBU(c, self.stepsizes[t])
                    
                    if con_DBU.no_of_boxes != 0:
                        
                        necc_tuple = con_DBU.dbu_to_tuple()
                        if necc_tuple not in all_condition_dicts:
                            all_condition_DBUs.append(con_DBU)
                            all_condition_dicts[necc_tuple] = 1
           
            
            S = self.MDP.state_spaces[t] 
            
            optimal_errors = [[], *optimal_errors]
            optimal_bellman_errors = [[], *optimal_bellman_errors]
            stored_DBUs = [[], *stored_DBUs]
            optimal_actions = [[], *optimal_actions]
            optimal_fixed_bellman_errors = [[], *optimal_fixed_bellman_errors]
            optimal_maxim_bellman_errors = [[], *optimal_maxim_bellman_errors]
            
            condition_DBUs = all_condition_DBUs
            optimal_cond_DBU = None

            for l in range(self.max_lengths[t]-1):
                
                min_error = np.inf
                optimal_action = None
                no_of_null_DBUs = 0
                
                print(f'We are at timestep {t} and lengthstep {l}')
                
                max_conditions = min(self.max_conditions[t], len(condition_DBUs))
                
                (optimal_cond_DBU, optimal_action, min_error, min_const_error,
                 min_complexity_error, optimal_bellman_error, error_df) = self.compute_errors_parallel(condition_DBUs, t, l, DBU, obs_states,
                                                                                                       integration_method, store_errors,
                                                                                                       conditions_string, max_workers=None)                      
                new_condition_DBUs = []                                         
                new_condition_dicts = {}                                       
                
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                total_error += min_error
                total_bellman_error += optimal_bellman_error
                
                for i, cond_dbu in enumerate(condition_DBUs):                       
                    sub_DBU = cond_dbu.subtract_DBUs(optimal_cond_DBU)          
                   
                    necc_tuple = sub_DBU.dbu_to_tuple()
                    if sub_DBU.no_of_boxes == 0:
                       	no_of_null_DBUs = no_of_null_DBUs + 1
                    elif necc_tuple not in new_condition_dicts:         
                        new_condition_dicts[necc_tuple] = 1             
                        new_condition_DBUs.append(sub_DBU)	         
                
                print(f'Size of new condition DBUs is {len(new_condition_DBUs)}')
                
                print(f'Timestep {t} and lengthstep {l}:')                      
                print('----------------------------------------------------------------')
                print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
                print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
                
                print(f'Optimal error is {min_error}')
                print(f'Optimal const error is {min_const_error}')
                print(f'Optimal complexity error is {min_complexity_error}')
                print(f'Optimal bellman error is {optimal_bellman_error}')
                
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Eta is {self.etas[t]}, Rho is {self.rhos[t]}')
     
                all_condition_dicts = new_condition_dicts                                                        
                condition_DBUs = new_condition_DBUs
                
                if len(optimal_errors) == 0:
                    optimal_errors = [[min_error]]
                else:
                    optimal_errors[0].append(min_error)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                
                if len(stored_DBUs) == 0:
                    stored_DBUs = [[optimal_cond_DBU]]
                else:
                    stored_DBUs[0].append(optimal_cond_DBU)

                if len(optimal_actions) == 0:
                    optimal_actions = [[optimal_action]]
                else:
                    optimal_actions[0].append(optimal_action)    

                if len(optimal_bellman_errors) == 0:
                    
                    optimal_bellman_errors = [[optimal_bellman_error]]
                    
                else:
                    optimal_bellman_errors[0].append(optimal_bellman_error)
                
                if (remaining_space.no_of_boxes == 0):
                    
                    print('--------------------------------------------------------------')
                    print(f'For timestep {t} we end at lengthstep {l}')

                    int_policy = VIDTR.get_interpretable_policy_dbus(stored_DBUs[0],
                                                                     optimal_actions[0])
                    
                    int_policies = [int_policy] + int_policies
                    
                    access_last_lengthstep = False
                    
                    if (t == self.MDP.time_horizon - 1):
                        
                        int_value_function = VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)
                        int_value_function = self.memoizer(t, int_value_function)                                
                        self.int_value_checks(t, int_value_function, int_policy)
                        
                        int_value_functions[t] = int_value_function
                        self.int_value_functions = int_value_functions
                        
                        
                    else:
                        
                        int_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                              self.MDP, int_value_functions[t+1], debug=debug)
                        
                        int_value_function = self.memoizer(t, int_value_function)
                        
                        self.int_value_checks(t, int_value_function, int_policy)
                        
                        int_value_functions[t] = int_value_function
                        self.int_value_functions = int_value_functions
                    
                    break
                    
            #Final lengthstep - We can only choose the optimal action here and we work over S - \cap_{i=1}^K S_i
            if access_last_lengthstep:
                min_error = np.inf
                
                bellman_map = bellman_equation(t, self.MDP, self.optimal_value_funcs)
                maxim_bellman_function = maximum_over_actions(t, bellman_map, self.MDP)
                for a in self.MDP.action_spaces[t]:
                    
                    int_bellman_map = bellman_equation_I(self.MDP, t, self.int_value_functions)
                    fixed_bellman_function = functools.partial(int_bellman_map,
                                                               a=a)
                    
                    const_eta_function = functools.partial(constant_map,
                                                           c = -self.etas[t])
                    
                    if integration_method == 'trajectory_integrate':
                        
                        
                        maxim_bellman_error = DBU.trajectory_integrate(obs_states,
                                                                       maxim_bellman_function,
                                                                       remaining_space,
                                                                       t)
                        
                        fixed_bellman_error = DBU.trajectory_integrate(obs_states,
                                                                       fixed_bellman_function,
                                                                       remaining_space,
                                                                       t)
                        
                        const_error = DBU.trajectory_integrate(obs_states,
                                                               const_eta_function,
                                                               remaining_space,
                                                               t)
                        
                        
                        error = maxim_bellman_error - fixed_bellman_error + const_error
                        
                    else:
                        
                        maxim_bellman_error = DBU.integrate_static(remaining_space,
                                                                   maxim_bellman_function)
                        
                        fixed_bellman_error = DBU.integrate_static(remaining_space,
                                                                   fixed_bellman_function)
                        
                        const_error = DBU.integrate_static(remaining_space,
                                                           const_eta_function)
                        
                        error = maxim_bellman_error - fixed_bellman_error + const_error
                    
                    
                    bellman_error = - fixed_bellman_error + maxim_bellman_error
                    if store_errors:
                        row = {
                                'Time': t,
                                'Length': l,
                                'Centre': remaining_space.centres,
                                'Lengths': remaining_space.lengths,
                                'Bellman_error': bellman_error,
                                'Action' : a,
                                'Constant_error': const_error,
                                'Complexity_error': 'None',
                                'Total_error': error,
                                'Integration_method' : integration_method,
                                'Conditions_string' : conditions_string
                            }     
                        
                        error_df = pd.concat([error_df, pd.DataFrame([row])], ignore_index=True)
                    
                    if error<min_error:
                        
                        optimal_action = a
                        min_error = error
                        optimal_maxim_bellman_error = maxim_bellman_error 
                        optimal_fixed_bellman_error = fixed_bellman_error
                        
                    total_error += min_error
                    total_bellman_error += min_error
                
                optimal_errors[0].append(min_error)
                optimal_actions[0].append(optimal_action)
                
                optimal_fixed_bellman_errors[0].append(optimal_fixed_bellman_error)
                optimal_maxim_bellman_errors[0].append(optimal_maxim_bellman_error)            
                
                int_policy = VIDTR.get_interpretable_policy_dbus(stored_DBUs[0],
                                                                 optimal_actions[0])
                
                int_policies = [int_policy] + int_policies
                
                if (t == self.MDP.time_horizon - 1):
                    
                    int_value_function = VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)
                
                    
                    int_value_function = self.memoizer(t, int_value_function)                                
                    self.int_value_checks(t, int_value_function, int_policy)
                    
                    int_value_functions[t] = int_value_function
                    self.int_value_functions = int_value_functions
                    
                    
                else:
                    
                    int_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                          self.MDP, int_value_functions[t+1], debug=debug)
                    
                    int_value_function = self.memoizer(t, int_value_function)
                    
                    self.int_value_checks(t, int_value_function, int_policy)
                    
                    int_value_functions[t] = int_value_function
                    self.int_value_functions = int_value_functions
            
            
        if store_errors:
            error_df.to_csv(error_file_name, index=False)
               
        self.optimal_errors = optimal_errors
        self.optimal_bellman_errors = optimal_bellman_errors
        self.optimal_maxim_bellman_errors = optimal_maxim_bellman_errors
        self.optimal_fixed_bellman_errors = optimal_fixed_bellman_errors
        self.optimal_actions = optimal_actions
        self.stored_DBUs = stored_DBUs
        self.total_bellman_error = total_bellman_error
        self.total_error = total_error
        self.int_policies = int_policies
        
        return stored_DBUs, optimal_actions
    
    @staticmethod
    def get_interpretable_policy_conditions(conditions, actions):
        '''                                                                    
        Given the conditions defining the policy, obtain the interpretable policy
        implied by the conditions.                                             
        
        Parameters:
        -----------------------------------------------------------------------
        conditions : np.array[l]
                     The conditions we want to represent in the int. policy             
        
        actions : np.array[l]
                  The actions represented in the int. policy
                                                                               
        '''

        def policy(state):
            
            for i, cond in enumerate(conditions):                               
                                                                                 
                if cond.contains_point(state):                                  
                                                                                
                    return actions[i]                                           
            
                                                                                
            return actions[-1]                                                  
        
        return policy
    
    @staticmethod
    def get_interpretable_policy_dbus(stored_dbus, actions):
        '''
        Given the dbus defining the policy, obtain the interpretable policy implied
        by the dbus.
        
        Parameters:
        -----------------------------------------------------------------------
        stored_dbus : list[DisjointBoxUnion]
                      The dbus we wish to derive an int. policy out of
        
        actions : list[np.array]
                  The actions we wish to represent in our int. policy
        
        '''
        def policy(state):
            
            for i, dbu in enumerate(stored_dbus):
                
                if dbu.is_point_in_DBU(state):
                    return actions[i]
            
            return actions[-1]
        
        return policy
    
    
    @staticmethod
    def tuplify_2D_array(two_d_array):
        two_d_list = []
        n,m = two_d_array.shape
        
        for i in range(n):
            two_d_list.append([])
            for j in range(m):
                two_d_list[-1].append(two_d_array[i,j])
        
            two_d_list[-1] = tuple(two_d_array[-1])                                 
        two_d_tuple = tuple(two_d_list)                                             
        return two_d_tuple                                                     
        
    
    def plot_errors(self):
        '''
        Plot the errors obtained after we perform the VIDTR algorithm.          

        '''
        for t in range(len(self.optimal_errors)):                                  
            plt.plot(np.arange(len(self.optimal_errors[t])), self.optimal_errors[t])
            plt.title(f'Errors at time {t}')
            plt.xlabel('Lengths')
            plt.ylabel('Errors')
            plt.show()
            
            plt.plot(np.arange(len(self.optimal_bellman_errors[t])), self.optimal_bellman_errors[t])
            plt.title(f'Bellman Errors at time {t}')
            plt.xlabel('Time')                                                 
            plt.ylabel('Bellman Error')                                         
            plt.show()
            
            plt.plot(np.arange(len(self.optimal_fixed_bellman_errors[t])), self.optimal_fixed_bellman_errors[t])
            plt.title(f'Fixed Bellman Errors at time {t}')
            plt.xlabel('Time')                                                 
            plt.ylabel('Fixed Bellman Error')                                         
            plt.show()

            plt.plot(np.arange(len(self.optimal_maxim_bellman_errors[t])), self.optimal_maxim_bellman_errors[t])
            plt.title(f'Maxim Bellman Errors at time {t}')
            plt.xlabel('Time')                                                 
            plt.ylabel('Maxim Bellman Error')                                         
            plt.show()
