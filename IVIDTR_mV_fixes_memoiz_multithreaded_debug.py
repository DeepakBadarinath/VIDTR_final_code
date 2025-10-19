#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:36:08 2024

@author: badarinath
"""

import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from importlib import reload
import inspect
import math
import concurrent.futures


import markov_decision_processes as mdp_module
import disjoint_box_union_parallelized                                                                  
import constraint_conditions as cc                                              
import pandas as pd
import functools
import random


import itertools
import sys
sys.setrecursionlimit(9000)  # Set a higher limit if needed
                                                                                           
#%%

mdp_module = reload(mdp_module)
disjoint_box_union_parallelized = reload(disjoint_box_union_parallelized)
cc = reload(cc)

from disjoint_box_union_parallelized import DisjointBoxUnion as DBU

integrate_static = DBU.integrate_static
trajectory_integrate = DBU.trajectory_integrate
easy_integral = DBU.easy_integral

from concurrent.futures import ProcessPoolExecutor, as_completed

from functools import lru_cache
import numpy as np
import types

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
    """
    Picklable equivalent of the nested bellman_map(s, a).
    """
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

        dbu_iter_class = disjoint_box_union_parallelized.DBUIterator(new_space)
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

    # choose integration method
    if integration_method == 'trajectory_integrate':
        maxim_bellman_error = trajectory_integrate(obs_states, maxim_bellman_function, cond_DBU, t)
        fixed_bellman_error = trajectory_integrate(obs_states, fixed_bellman_function, cond_DBU, t)
        const_error = trajectory_integrate(obs_states, eta_function, cond_DBU, t)
    else:
        maxim_bellman_error = integrate_static(
            cond_DBU,
            maxim_bellman_function,
            check_positivity=True,
            function_name=f'Maxim_Bellman_function_time_{t}_cond_no_{i}_action_{act_no}'
        )
        
        fixed_bellman_error = integrate_static(
            cond_DBU,
            fixed_bellman_function,
            check_positivity=True,
            function_name=f'Fixed_Bellman_function_time_{t}_cond_no_{i}_action_{act_no}'
        )
        
        const_error = integrate_static(cond_DBU, eta_function)

    complexity_error = rho * cond_DBU.complexity * cond_DBU.no_of_points()
    bellman_error = maxim_bellman_error - fixed_bellman_error
    total_error = bellman_error - const_error + complexity_error
    
    
    result_row = None
    if store_errors:
        result_row = {
            'Time': t,
            'Length': l,
            'Centre': cond_DBU.centres,
            'Lengths': cond_DBU.lengths,
            'Maxim_bellman_error': maxim_bellman_error,
            'Fixed_bellman_error': fixed_bellman_error,
            'Bellman_error' : bellman_error,
            'Constant_error': const_error,
            'Action': a,
            'Complexity_error': complexity_error,
            'Total_error': total_error,
            'Integration_method': integration_method,
            'Conditions_string': conditions_string
        }

    return (total_error, const_error, complexity_error, bellman_error, result_row)


# Parallelize subtraction of DBUs
def subtract_and_filter(cond_dbu, optimal_cond_DBU):
    sub_DBU = cond_dbu.subtract_DBUs(optimal_cond_DBU)
    if sub_DBU.no_of_boxes == 0:
        return None
    return (sub_DBU.dbu_to_tuple(), sub_DBU)

# Reverse a list l
def rev(l):
    l.reverse()
    return l

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
         
         dbu_iter_class = disjoint_box_union_parallelized.DBUIterator(self.MDP.state_spaces[t])
         
         state_iterator = iter(dbu_iter_class)
         
         f_dict = {}
         
         for s in state_iterator:
             
             f_dict[tuple(s)] = f(s)
         
         f_memoized = convert_dict_to_function(f_dict, state_iterator)
         return f_memoized
    
    
    def compute_optimal_policies_mpi(self, comm=None, use_lru=True):
        
        """
        MPI-capable version of compute_optimal_policies().
        If mpi4py is available and comm.size > 1, this distributes the per-state
        Bellman backups across ranks. Otherwise runs a single-process version.
    
        Arguments:
        -----------------------------------------------------------------------
        - comm: MPI communicator (optional). If None, tries to import mpi4py and use MPI.COMM_WORLD.
        - use_lru: whether to lru_cache the bellman_map on each rank (recommended).
    
        Returns:
        -----------------------------------------------------------------------
        - optimal_policy_funcs, optimal_value_funcs as lists of callables (same as original).
        
        """
        # ----- Setup MPI communicator (optional) -----
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except Exception:
                comm = None
    
        is_distributed = (comm is not None) and (comm.Get_size() > 1)
        if is_distributed:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
    
        # ----- Helper: chunk indices (deterministic partitioning) -----
        def chunk_indices(n, size, rank):
            per = n // size
            rem = n % size
            start = rank * per + min(rank, rem)
            end = start + per + (1 if rank < rem else 0)
            return start, end
    
        # ----- Precompute deterministic state lists (same on every rank) -----
        # This avoids re-instantiating DBUIterator repeatedly in inner loops.
        T = self.MDP.time_horizon
        state_lists = []
        for t in range(T):
            dbu_iter = disjoint_box_union_parallelized.DBUIterator(self.MDP.state_spaces[t])
            # convert states to immutable tuple keys once and keep global order
            states = [tuple(s) for s in dbu_iter]
            state_lists.append(states)
    
        # ----- Initialize containers (dicts per timestep) -----
        optimal_value_dicts = [None] * T
        optimal_policy_dicts = [None] * T
    
        # Preinitialize value functions to zero (as your original code did)
        zero_func = functools.partial(constant_map, c=0)
        self.optimal_policy_funcs = [zero_func for _ in range(T)]
        self.optimal_value_funcs = [zero_func for _ in range(T)]
    
        # ----- Main backward DP loop (t = T-1 ... 0) -----
        for t in range(T - 1, -1, -1):
            states = state_lists[t]
            nstates = len(states)
            start, end = chunk_indices(nstates, size, rank)
            local_states = states[start:end]
    
            # Prebuild bellman_map once for this t using your existing function
            # bellman_equation(t, self.MDP, self.optimal_value_funcs) should return callable f(s, a)
            bellman_map = bellman_equation(t, self.MDP, self.optimal_value_funcs)
    
            # Wrap and cache bellman_map to accept tuple keys and avoid repeated array conversions.
            if use_lru:
                @functools.lru_cache(maxsize=None)
                def bellman_cached(s_key, a_idx):
                    # s_key is tuple; a_idx is index into action list
                    s_arr = np.array(s_key)
                    a = self.MDP.action_spaces[t][a_idx]
                    return float(bellman_map(s_arr, a))
            else:
                def bellman_cached(s_key, a_idx):
                    s_arr = np.array(s_key)
                    a = self.MDP.action_spaces[t][a_idx]
                    return float(bellman_map(s_arr, a))
    
            actions = self.MDP.action_spaces[t]
            nacts = len(actions)
    
            # ----- Local compute: evaluate best action/value for each local state -----
            local_vals = np.empty(len(local_states), dtype=np.float64)
            local_acts_idx = np.empty(len(local_states), dtype=np.int32)
    
            for i, s_key in enumerate(local_states):
                best_v = -np.inf
                best_ai = 0
                for ai in range(nacts):
                    v = bellman_cached(s_key, ai)
                    if v > best_v:
                        best_v = v
                        best_ai = ai
                local_vals[i] = best_v
                local_acts_idx[i] = best_ai
    
            # ----- Gather pieces across ranks to form the full arrays of length nstates -----
            if is_distributed:
                # 1) gather counts from every rank
                local_n = np.array([local_vals.size], dtype='i')
                counts = np.empty(size, dtype='i')
                comm.Allgather([local_n, MPI.INT], [counts, MPI.INT])
    
                # 2) compute displacements
                displs = np.zeros_like(counts)
                displs[1:] = np.cumsum(counts)[:-1]
    
                total_n = int(counts.sum())
                # sanity
                if total_n != nstates:
                    # If mismatch, something wrong with deterministic partitioning
                    if rank == 0:
                        raise RuntimeError(f"Total gathered states {total_n} != expected {nstates}")
                    else:
                        comm.Barrier()
                        return None
    
                # 3) prepare recv buffers
                full_vals = np.empty(total_n, dtype=np.float64)
                full_acts = np.empty(total_n, dtype=np.int32)
    
                # 4) Allgatherv local arrays into full arrays
                comm.Allgatherv([local_vals, MPI.DOUBLE],
                                [full_vals, (counts, displs), MPI.DOUBLE])
                comm.Allgatherv([local_acts_idx, MPI.INT],
                                [full_acts, (counts, displs), MPI.INT])
            else:
                # single-process: full arrays are simply local arrays placed at correct positions
                full_vals = np.empty(nstates, dtype=np.float64)
                full_acts = np.empty(nstates, dtype=np.int32)
                full_vals[start:end] = local_vals
                full_acts[start:end] = local_acts_idx
    
            # ----- Build dictionaries in the global deterministic order -----
            val_dict = {}
            policy_dict = {}
            # full_vals and full_acts are in the same order as `states`
            for idx, s_key in enumerate(states):
                val_dict[s_key] = float(full_vals[idx])
                policy_dict[s_key] = actions[int(full_acts[idx])]
    
            optimal_value_dicts[t] = val_dict
            optimal_policy_dicts[t] = policy_dict
    
            # ----- Convert dicts to callable functions (same semantics as original) -----
            # convert_dict_to_function expects an iterator of states; we can supply iter(states)
            optimal_policy_func = convert_dict_to_function(policy_dict, iter(states),
                                                           default_value=actions[0])
            optimal_value_func = convert_dict_to_function(val_dict, iter(states),
                                                          default_value=0.0)
    
            # store into object so next timestep's bellman_equation uses them
            self.optimal_policy_funcs[t] = optimal_policy_func
            self.optimal_value_funcs[t] = optimal_value_func
    
            # clear lru cache to free memory
            if use_lru:
                bellman_cached.cache_clear()
    
        # ----- Finalize: return same outputs as original function -----
        # Build the lists (already stored in self.*)
        optimal_policy_funcs = self.optimal_policy_funcs
        optimal_value_funcs = self.optimal_value_funcs
    
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
            
            dbu_iter_class = disjoint_box_union_parallelized.DBUIterator(new_space)              
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
                                                                                
        dbu_iter_class = disjoint_box_union_parallelized.DBUIterator(iter_space)              
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
        
        dbu_iter_class = disjoint_box_union_parallelized.DBUIterator(self.MDP.state_spaces[t])
        state_iterator = iter(dbu_iter_class)
        
        for s in state_iterator:
            
            int_value = int_value_function(s)
            optimal_value = self.optimal_value_funcs[t](s)
            
            if int_value > optimal_value:
                
                optimal_value = self.optimal_value_funcs[t](s)
                raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")
    

   
    def compute_errors_parallel(self, condition_DBUs, t, l, DBU, obs_states,
                                integration_method, store_errors,
                                conditions_string, max_workers=None):

        # Prepare all (action, cond_DBU) tasks in one go
        tasks = []
        function_dicts = {}
        int_bellman_map = bellman_equation_I(self.MDP, t, self.int_value_functions)
        
       
        bellman_map = bellman_equation(t, self.MDP, self.optimal_value_funcs)
        maxim_bellman_function = maximum_over_actions(t, bellman_map, self.MDP)
       
        neg_const_eta = functools.partial(constant_map,
                                          c = -self.etas[t])
        
        
        for i, cond_DBU in enumerate(condition_DBUs):
            # Bind the Bellman function to this action
            
            for act_no, a in enumerate(self.MDP.action_spaces[t]):
                
                fixed_bellman = functools.partial(int_bellman_map, a=a)
                function_dicts = {
                    'fixed_bellman_function': fixed_bellman,
                    'eta_function': neg_const_eta,
                    'maxim_bellman_function' : maxim_bellman_function
                    }
                
                tasks.append((i, cond_DBU, act_no, a, t, l, self.rhos[t], obs_states,
                              function_dicts, integration_method, store_errors, conditions_string))
    
        # Execute all tasks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_error_for_cond_action, args) for args in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
    
        # Each result is (total_err, const_err, complexity_err, bellman_err, result_row)
        # Now find the global minimum over all actions and cond_DBUs
        min_index, min_error_tuple = min(enumerate(results), key=lambda x: x[1][0])                                                                                                                                                 
    
        min_total_error, min_const_error, min_complexity_error, min_bellman_error, result_row = min_error_tuple                                                    
        print(f"✅ Global minimum total error = {min_total_error:.6f}")
    
        # Retrieve associated optimal action and cond_DBU from tasks[min_index]
        _, cond_DBU, act_no, a, _, _, _, _, _, _, _, _ = tasks[min_index]
        optimal_action = a
    
        print(f"Optimal action: {optimal_action}")
    
        # Reconstruct optimal cond_DBU from result_row
        
        dbu_dim = len(np.array(result_row['Centre']).flatten())
        centre = np.array(result_row['Centre']).reshape(1, -1)
        length = np.array(result_row['Lengths']).reshape(1, -1)
        optimal_cond_DBU = DBU(1, dbu_dim, length, centre)
    
        print('For optimal cond_DBU we have')
        print(f'Dimension is {dbu_dim}')
        print(f'Centre is {centre}')    
        print(f'Length is {length}')
        print(optimal_cond_DBU)    
    
        #  no_of_boxes, dimension, lengths, centres,
    
        # Optionally store error DataFrame
        error_df = None
        if store_errors:
            valid_rows = [r[-1] for r in results if isinstance(r[-1], dict) or isinstance(r[-1], pd.Series)]
            error_df = pd.DataFrame(valid_rows)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    
        return (optimal_cond_DBU, optimal_action,
                min_total_error, min_const_error,
                min_complexity_error, min_bellman_error, error_df)
        
    
    def compute_interpretable_policies(self,
                                       integration_method='trajectory_integrate',
                                       integral_percent=0.5, debug=False,
                                       obs_states=None, conditions_string='all',
                                       store_errors=False,
                                       error_file_name='error_df.csv'):
        """
        Compute interpretable policies via dynamic programming over Disjoint Box Unions (DBUs).

        Parameters:
        -----------------------------------------------------------------------
        integration_method : str, default='trajectory_integrate'
            Integration method to use for computing Bellman errors. Options include
            'trajectory_integrate', 'integrate_static', or 'easy_integral'.

        integral_percent : float, default=0.5
            Percentage used for approximating integrals within the DBU integration routines.

        debug : bool, default=False
            If True, prints detailed intermediate information during computation.

        obs_states : list or None, default=None
            List of observed trajectories (states) used for integration and policy evaluation.
            Each entry corresponds to a trajectory consisting of states over all timesteps.

        conditions_string : str, default='all'
            Specifies how to generate condition sets at each timestep. Accepted values:
                - 'all' : generate all possible conditions up to max complexity
                - 'order_stats' : generate conditions based on order statistics of observed states

        store_errors : bool, default=False
            Whether to store the computed error DataFrames for each timestep and lengthstep.

        error_file_name : str, default='error_df.csv'
            Filename for saving the aggregated error DataFrame (only used if store_errors=True).

        Returns:
        -----------------------------------------------------------------------
        stored_DBUs : list
            A list of lists of DBUs chosen at each timestep and lengthstep.

        optimal_actions : list
            A list of lists of optimal actions corresponding to the stored DBUs.

        Notes
        -----
        This method iterates backward in time to compute interpretable policies using a form of
        dynamic programming that builds interpretable tree-like structures over DBUs. For each
        timestep and lengthstep, it selects the condition-action pair minimizing the combined
        Bellman and complexity errors, parallelizing heavy DBU subtraction steps for efficiency.
        """

        MDP = self.MDP
        time_horizon = MDP.time_horizon
        max_complexity = self.max_complexity
        max_lengths = self.max_lengths
        max_conditions = self.max_conditions
        etas = self.etas
        rhos = self.rhos
        stepsizes = self.stepsizes

        zero_func = functools.partial(constant_map, c=0)
        int_value_functions = [zero_func for _ in range(time_horizon)]
        self.int_value_functions = int_value_functions

        collected_stored_DBUs = []
        collected_optimal_actions = []
        collected_optimal_bellman_errors = []
        collected_optimal_errors = []
        collected_opt_fixed_bellman_errors = []
        collected_opt_maxim_bellman_errors = []
        collected_int_policies = []

        total_error = 0.0
        total_bellman_error = 0.0

        error_rows = []

        def dbg(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        for t in range(time_horizon - 1, -1, -1):
            dbg(f"Starting timestep {t}")

            if conditions_string == 'all':
                all_conditions = MDP.state_spaces[t].generate_all_conditions(max_complexity[t])
            elif conditions_string == 'order_stats':
                obs_states_at_t = [obs_states[i][t] for i in range(len(obs_states))]
                all_conditions = MDP.state_spaces[t].generate_conditions_from_observations(obs_states_at_t, max_complexity[t])
            else:
                all_conditions = []

            remaining_space = MDP.state_spaces[t]

            cond_dbu_by_tuple = {}
            for c in all_conditions:
                if c is None:
                    continue
                con_DBU = DBU.condition_to_DBU(c, stepsizes[t])
                if con_DBU.no_of_boxes == 0:
                    continue
                key = con_DBU.dbu_to_tuple()
                if key not in cond_dbu_by_tuple:
                    cond_dbu_by_tuple[key] = con_DBU
            condition_DBUs = list(cond_dbu_by_tuple.values())

            time_stored_DBUs = []
            time_opt_actions = []
            time_opt_errors = []
            time_opt_bellman_errors = []
            time_opt_fixed_bellman_errors = []
            time_opt_maxim_bellman_errors = []

            access_last_lengthstep = True

            for l in range(max_lengths[t] - 1):
                dbg(f'At timestep {t}, lengthstep {l} - conditions: {len(condition_DBUs)}')

                if not condition_DBUs:
                    dbg('No condition DBUs left, breaking')
                    break

                (optimal_cond_DBU, optimal_action, min_error, min_const_error,
                 min_complexity_error, optimal_bellman_error, error_df_t_l) = (
                    self.compute_errors_parallel(condition_DBUs, t, l, DBU, obs_states,
                                                   integration_method, store_errors,
                                                   conditions_string, max_workers=None)
                )

                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                total_error += min_error
                total_bellman_error += optimal_bellman_error
          
                new_cond_by_tuple = {}
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # repeat optimal_cond_DBU so each worker gets (cond_dbu, optimal_cond_DBU)
                    for result in executor.map(subtract_and_filter, condition_DBUs, itertools.repeat(optimal_cond_DBU)):
                        if result is None:
                            continue
                        key, sub_DBU = result
                        if key not in new_cond_by_tuple:
                            new_cond_by_tuple[key] = sub_DBU
                            condition_DBUs = list(new_cond_by_tuple.values())

                dbg(f'Size of new condition DBUs is {len(condition_DBUs)}')
                dbg('----------------------------------------------------------------')
                dbg(f'Optimal action at timestep {t}, lengthstep {l}: {optimal_action}')
                dbg(f'Optimal conditional DBU: {optimal_cond_DBU}')
                dbg(f'Optimal error: {min_error} | const: {min_const_error} | complexity: {min_complexity_error} | bellman: {optimal_bellman_error}')
                dbg(f'Eta = {etas[t]}, Rho = {rhos[t]}')

                time_opt_errors.append(min_error)
                time_stored_DBUs.append(optimal_cond_DBU)
                time_opt_actions.append(optimal_action)
                time_opt_bellman_errors.append(optimal_bellman_error)

                if store_errors and error_df_t_l is not None:
                    error_rows.append(error_df_t_l)

                if remaining_space.no_of_boxes == 0:
                    dbg(f'For timestep {t} we end at lengthstep {l}')
                    int_policy = VIDTR.get_interpretable_policy_dbus(time_stored_DBUs, time_opt_actions)
                    collected_int_policies.append(int_policy)
                    access_last_lengthstep = False

                    if t == time_horizon - 1:
                        int_value_function = VIDTR.last_step_int_value_function(t, int_policy, MDP, debug=debug)
                    else:
                        int_value_function = VIDTR.general_int_value_function(t, int_policy, MDP, int_value_functions[t + 1], debug=debug)

                    int_value_function = self.memoizer(t, int_value_function)
                    self.int_value_checks(t, int_value_function, int_policy)
                    int_value_functions[t] = int_value_function
                    self.int_value_functions = int_value_functions

                    break

            if access_last_lengthstep:
                min_error = np.inf
                optimal_action = None
                optimal_fixed_bellman_error = None
                optimal_maxim_bellman_error = None

                maxim_bellman_map = maximum_over_actions(t, bellman_equation(t, MDP, self.optimal_value_funcs), MDP)

                for a in MDP.action_spaces[t]:
                    int_bellman_map = bellman_equation_I(MDP, t, int_value_functions)
                    fixed_bellman_function = functools.partial(int_bellman_map, a=a)
                    const_eta_function = functools.partial(constant_map, c=-etas[t])

                    if integration_method == 'trajectory_integrate':
                        maxim_bellman_error = DBU.trajectory_integrate(obs_states, maxim_bellman_map, remaining_space, t)
                        fixed_bellman_error = DBU.trajectory_integrate(obs_states, fixed_bellman_function, remaining_space, t)
                        const_error = DBU.trajectory_integrate(obs_states, const_eta_function, remaining_space, t)
                        error = maxim_bellman_error - fixed_bellman_error + const_error
                    else:
                        maxim_bellman_error = DBU.integrate_static(remaining_space, maxim_bellman_map)
                        fixed_bellman_error = DBU.integrate_static(remaining_space, fixed_bellman_function)
                        const_error = DBU.integrate_static(remaining_space, const_eta_function)
                        error = maxim_bellman_error - fixed_bellman_error - const_error

                    if error < min_error:
                        optimal_action = a
                        min_error = error
                        optimal_maxim_bellman_error = maxim_bellman_error
                        optimal_fixed_bellman_error = fixed_bellman_error

                time_opt_errors.append(min_error)
                time_opt_actions.append(optimal_action)
                time_stored_DBUs.append(remaining_space)
                time_opt_bellman_errors.append(min_error)
                time_opt_fixed_bellman_errors.append(optimal_fixed_bellman_error)
                time_opt_maxim_bellman_errors.append(optimal_maxim_bellman_error)

                int_policy = VIDTR.get_interpretable_policy_dbus(time_stored_DBUs, time_opt_actions)
                collected_int_policies.append(int_policy)

                if t == time_horizon - 1:
                    int_value_function = VIDTR.last_step_int_value_function(t, int_policy, MDP, debug=debug)
                else:
                    int_value_function = VIDTR.general_int_value_function(t, int_policy, MDP, int_value_functions[t + 1], debug=debug)

                int_value_function = self.memoizer(t, int_value_function)
                self.int_value_checks(t, int_value_function, int_policy)
                int_value_functions[t] = int_value_function
                self.int_value_functions = int_value_functions

            collected_stored_DBUs.append(time_stored_DBUs)
            collected_optimal_actions.append(time_opt_actions)
            collected_optimal_errors.append(time_opt_errors)
            collected_optimal_bellman_errors.append(time_opt_bellman_errors)
            collected_opt_fixed_bellman_errors.append(time_opt_fixed_bellman_errors)
            collected_opt_maxim_bellman_errors.append(time_opt_maxim_bellman_errors)

        if store_errors and error_rows:
            error_df = pd.concat(error_rows, ignore_index=True)
            error_df.to_csv(error_file_name, index=False)


        self.optimal_errors = rev(collected_optimal_errors)
        self.optimal_bellman_errors = rev(collected_optimal_bellman_errors)
        self.optimal_maxim_bellman_errors = rev(collected_opt_maxim_bellman_errors)
        self.optimal_fixed_bellman_errors = rev(collected_opt_fixed_bellman_errors)
        self.optimal_actions = rev(collected_optimal_actions)
        self.stored_DBUs = rev(collected_stored_DBUs)
        self.total_bellman_error = total_bellman_error
        self.total_error = total_error
        self.int_policies = rev(collected_int_policies)

        return self.stored_DBUs, self.optimal_actions

    
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
