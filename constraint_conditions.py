#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:25:07 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

#%%
class ConstraintConditions: 
    
    '''
    A constraint condition for the VIDTR method; data structure represents:
        x_i0 \in [a_i0, b_i0) \times x_i1 \in [a_i1, b_i1) \times ... x_ik \in \times [a_ik, b_ik)
    '''
    def __init__(self, dimension, non_zero_indices,
                 bounds, state_bounds = []):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimension : int                                                        
                    The dimension of the space
                    
        non_zero_indices : np.array[complexity]
                           The non-trivial indices in the constraint condition
        
        bounds : np.array[complexity, 2]
                 The values to be indicated in the constraints
        
        state_bounds : np.array[dimension, 2]
                       The bounds of the state space the condition is assumed to
                       be in
        
        Stores:
        -----------------------------------------------------------------------
        complexity : int
                     The number of non-zero elements in the greater than array
        '''
        
        self.dimension = dimension
        self.state_bounds = state_bounds
        
        self.complexity = len(non_zero_indices)
        
        if len(state_bounds) > 0:
            
            state_bounds = np.array(state_bounds)
            
            constraint_bounds = np.array([[state_bounds[d,0], state_bounds[d,1]]
                                          for d in range(dimension)])           
            
            
            for i, ind in enumerate(non_zero_indices):
                
                if bounds[i,0] != -np.inf:
                    constraint_bounds[ind, 0] = bounds[i, 0]
                
                if bounds[i,1] != np.inf:
                    constraint_bounds[ind, 1] = bounds[i, 1]
                
            self.bounds = constraint_bounds
        else:
            self.bounds = bounds
        
        self.non_zero_indices = non_zero_indices
        
    def plot_2D_constraints(self, ax=None, show_plot=True, c = 'black',
                            x_bounds = None, y_bounds = None,
                            x_bounds_default = (-10,10), y_bounds_default = (-10,10),
                            title = 'Constraints'):
        '''
        Given a constraint condition, plot the dotted lines representing the 
        various constraints in 2D space.

        Parameters:
        -----------------------------------------------------------------------
        ax : plot
             The given plot
        
        show_plot : Boolean
                    A binary variable that determines whether we show the plot
        
        c : string
            String that encodes the color of the lines plotted
        
        x_bounds : Tuple
                   The x axis bounds on the horizontal line
        
        y_bounds : Tuple
                   The y axis bounds on the vertical line
        
        x_bounds_default : Tuple
                           The default x bounds when we do not have a plot
        
        y_bounds_default : Tuple
                           The default y bounds when we do not have a plot
        
        title : string
                The title of the plot 
                           
        Returns:
        -----------------------------------------------------------------------
        ax_new : 2D constraint plotted with dotted lines 
                 The final plot
        '''
        if ax == None:
            fig,ax = plt.subplots()
            if x_bounds == None:
                x_bounds = x_bounds_default
            if y_bounds == None:
                y_bounds = y_bounds_default
        
        else:
            x_bounds = ax.get_xlim()
            y_bounds = ax.get_ylim()

        ax.set_title(title)
        
        bounds = self.bounds
        if (self.bounds.shape == (1,2)):
            bounds = np.vstack([self.bounds, [-np.inf, np.inf]])
        
        
        if bounds[0,0] != -np.inf:
            ax.hlines(y=bounds[0,0], xmin=x_bounds[0], xmax=x_bounds[1],
                      linestyle='dotted', color=c)
        
        if bounds[0,1] != +np.inf:
            ax.hlines(y=bounds[0,1], xmin=x_bounds[0], xmax=x_bounds[1],
                      linestyle='dotted', color=c)
        
        if bounds[1,0] != -np.inf:
            ax.vlines(x=bounds[1,0], ymin=y_bounds[0], ymax=y_bounds[1],
                      linestyle='dotted', color=c)
        
        if bounds[1,1] != +np.inf:
            ax.vlines(x=bounds[1,1], ymin=y_bounds[0], ymax=y_bounds[1],
                      linestyle='dotted', color=c)
        
        if show_plot:
            plt.show()
        
        return ax
    
    def contains_point(self, point):
        '''
        Given a point in self.dimension space, check if the condition contains this 
        point.
        
        Parameters:
        -----------------------------------------------------------------------
        point : np.array(self.dimension)
                The point which we wish to check if it lies in the condition
        '''
        
        if len(self.state_bounds) > 0:
            for d in range(self.dimension):
                
                if self.bounds[d,0] > point[d] or point[d] >= self.bounds[d,1]:
                    return False
        
        else:
            for i, ind in enumerate(self.non_zero_indices):
                
                if self.bounds[i,0] > point[ind] or point[ind] >= self.bounds[i,1]:
                    return False
        
        return True


    def __str__(self):
        '''
        Prints an element from the ConstraintConditions class.
        '''
        final_str = ''
        
        if len(self.state_bounds) == 0:
            for i,ind in enumerate(self.non_zero_indices):
                final_str += f'{self.bounds[i,0]} <= x_{ind} < {self.bounds[i,1]}\n'
            
        else:
            for d in range(self.dimension):
                final_str += f'{self.bounds[d,0]} <= x_{d} < {self.bounds[d,1]}\n'

        return final_str
    
#%%

if __name__ == '__main__':
    dimension = 2
    indices = np.array([0,1])
    bounds = np.array([[-2, 4], [4, 5]])
    #state_bounds = np.array([[], []])
    
    constraint = ConstraintConditions(dimension, indices, bounds)
        
    print(constraint)
    constraint.plot_2D_constraints()
