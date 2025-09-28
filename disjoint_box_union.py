#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To dos are also listed in this .py file
"""                    
Created on Fri May 24 22:29:40 2024
                       
@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import constraint_conditions as cc                                             
import interval_and_rectangle_operations as op
import random
import time
from itertools import combinations, product
import constraint_conditions as cc
    
from importlib import reload
import itertools                                  
from itertools import combinations, product                                    
                                
from scipy import integrate
                                                
op = reload(op)
#%%                                                                                                                                                            
# Task breakdown:-                                                              
###############################################################################

# Baum Welch and other algorithms to get the models for the data - design and debug              
# VIDTR based on the examples from the IDTR paper                                    
                                                                                
# VIDTR based on the real world data                                             
###############################################################################         

#%%                                                                            
class DisjointBoxUnion:                                                         
                                                                                                                                                    
    '''                                                                        
    Represent a disjoint union of d-dimensional boxes in \mathbb{R}^d.         
    '''

    def __init__(self, no_of_boxes, dimension, lengths, centres, stepsizes = [],
                 store_endpoints = True, store_bounds = True, complexity = 0.0):
        '''
        Parameters:    
        -----------------------------------------------------------------------
            no_of_boxes : int                                                  
                          Number of disjoint boxes                                 

            dimension : int                                                    
                        Dimension of the space we reside in                    

            lengths : np.array(no_of_boxes, dimension)                          
                      A numpy array that denotes the lengths of the different  
                      dimensions                                               
                                                                               
            centres : np.array(no_of_boxes, dimension)                          
                      A numpy array denoting the coordinates of the different  
                      centres of the many boxes                                  
                                                                               
            stepsizes : np.array(no_of_boxes, dimension) or int or float       
                        stepsizes[i, d] = the stepsize for the nth box and the 
                        dth dimension
            
        complexity : int                                                       
                     The complexity of the DBU in case it is of type cond_DBU   
            
        Stores:
        -----------------------------------------------------------------------
            bounds : list(2 * d) of length no_of_boxes
                     The lower and upper bound for each dimension and each box
            
        end_points : np.array(N * 2D * D)
                     The end points of the n'th box each of which has d 
                     coordinates
                        
        '''

        self.no_of_boxes = no_of_boxes
        self.dimension = dimension
        
        if len(centres.shape) == 1:
            self.centres = np.expand_dims(centres, axis=0)
        else:
            self.centres = centres
        
        if len(lengths.shape) == 1:
            self.lengths = np.expand_dims(lengths, axis=0)
        else:
            self.lengths = lengths
        
        self.point_count = None
        
        if type(stepsizes) == int or type(stepsizes) == float:
            self.stepsizes = stepsizes * np.ones((no_of_boxes, dimension))
        
        elif np.sum(stepsizes) == 0:
            self.stepsizes = np.ones((no_of_boxes, dimension))
        
        elif len(stepsizes.shape) == 1:
            self.stepsizes = np.expand_dims(stepsizes, axis=0)
        
        else:
            self.stepsizes = stepsizes
        
        if store_bounds:
            self.bounds = self.get_bounds()
        if store_endpoints:
            self.endpoints = self.get_end_points()
        
        self.complexity = complexity
        self.all_points = []
    
    def no_of_points(self):
        '''
        Given a DBU return the total number of points in it
        '''
        point_count = 0
        for box_no in range(self.no_of_boxes):
            lengths = [DBUIterator.count_points(
                self.bounds[box_no][d][0],
                self.bounds[box_no][d][1] - 1e-12,
                self.stepsizes[box_no, d]) for
                d in range(self.dimension)]

            point_per_box = 1
            for l in lengths:
                point_per_box = point_per_box * l

            point_count += point_per_box
        
        if point_count == 0:
            point_count = 1
        
        return point_count
            
    
    def add_disjoint_rectangle(self, dimension, centre, lengths, stepsize=[]):
        '''
        Given a DBU add a new rectangle of the same dimension assuming it does
        not intersect with the existing DBU.
        
        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the new rectangle
        
        centre : np.array[dimension]
                 The centre of the new rectangle
        
        lengths : np.array[dimension]
                  The lengths of the sides of the new rectangle
        
        stepsize : np.array[dimension]
                   The stepsizes over the various dimensions of the new rectangle.
        
        '''
        
        if dimension != self.dimension:
            print(f'Dimension of new box which is {dimension} != {self.dimension} which is the dimension of the DBU.')
            return
        
        if self.no_of_boxes == 0:
            self.lengths = np.expand_dims(lengths, axis=0)
            self.stepsize = np.expand_dims(np.ones(dimension), axis=0)
            self.centres = np.expand_dims(centre, axis=0)
        
        else:
            self.lengths = np.vstack([self.lengths, lengths])
            stepsize = np.ones(dimension)
            self.stepsizes = np.vstack([self.stepsizes, stepsize])
            self.centres = np.vstack([self.centres, centre])

        self.no_of_boxes += 1        
    
    
    def sampling_integrate(self, function,
                           point_percentages = 0.5):
        '''
        Integrate a function over a disjoint box union data structure. We assume
        to do a MCMC based sampling integral by sampling points_per_box number of points
        or point_percentage number of points in each box.
        
        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d dimensional input function to integrate over the DBU.
        
            
            point_percentages : float or list
                               The percentage of points we sample from each 
                               box of the DBU to perform the integration of the function
        
        Returns:
        -----------------------------------------------------------------------
            integral : float
                       The estimate of the Riemannian summation integral over the DBU. 
        
        '''
        
        if type(point_percentages) == float:
            point_percentages = [point_percentages for i in range(self.no_of_boxes)]
        
        
        sampled_points = 0
        function_sum = 0
        for box_no in range(self.no_of_boxes):
            
            lengths = [DBUIterator.count_points(
                                   self.bounds[box_no][d][0],
                                   self.bounds[box_no][d][1] - 1e-12,
                                   self.stepsizes[box_no, d]) for
                                   d in range(self.dimension)]
            
            points_per_box = 1
            for l in lengths:
                points_per_box = points_per_box * l
        
            sampling_number = int(points_per_box * point_percentages[box_no])
            sampled_points += sampling_number
            
            for s in range(sampling_number):
                
                function_sum += function(self.sample_from_box(box_no))
                    
        
        return function_sum / sampled_points
    
    
    
    def integrate(self, function):
        '''
        Given a function taking inputs in d dimensions, evaluate the Riemann-Integral 
        of the function over the DBU.

        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d-dimensional function to integrate over the DBU
                     
        Returns:         
        -----------------------------------------------------------------------
            integral : float
                  The value of the integral of the function over the DBU.
                  
        '''
        
        if self.bounds == None:
            self.get_bounds()
    
        f_sum = 0.0
        
        
        for box_no in range(self.no_of_boxes):
            coord_list = []
            for dim in range(self.dimension):
                
                coord_list.append(np.arange(self.bounds[box_no][dim][0],
                                            self.bounds[box_no][dim][1] - 1e-12,
                                            self.stepsizes[box_no, dim]))         
        
            for point in itertools.product(*coord_list):
                f_sum = function(np.array(point)) + f_sum
            
        return f_sum

       
    def subtract_constraint(self, constraint_condition):
        '''
        Given a DBU subtract a constraint condition from it to give another DBU.

        Parameters:
        -----------------------------------------------------------------------
            constraint_condition : type(constraint_conditions)
                                   The constraint condition we wish to subtract
                                   from the DBU

        Returns:
        -----------------------------------------------------------------------
            final_DBU : The final disjoint box union after removing the region
            under the constraint condition
        '''
        final_DBU = DisjointBoxUnion(0, self.dimension,
                                     np.array([[]]), np.array([[]]))
        
        for box_no in range(self.no_of_boxes):
            
            constraint_DBU = DisjointBoxUnion.condition_to_DBU(constraint_condition,
                                                               self.stepsizes[box_no])
            
            single_box_DBU = DisjointBoxUnion(1, self.dimension,
                                              np.array([self.lengths[box_no]]),
                                              np.array([self.centres[box_no]]),
                                              np.array([self.stepsizes[box_no]]))
            
            subtracted_DBU = single_box_DBU.subtract_single_DBUs(constraint_DBU)
            if np.sum(subtracted_DBU.lengths) > 0:
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)

        return final_DBU
    
    
    def present_inside(self, new_DBU):
        
        '''
        Given a new_DBU, check if new_DBU is present inside the self DBU
        
        Parameters:
        -----------------------------------------------------------------------
        new_DBU : DisjointBoxUnion
                  Check if new_DBU is present inside the self DBU
        
        '''
        if self.subtract_DBUs(new_DBU).no_of_boxes == 0:
            return True
        else:
            return False
    
    def intersect_DBUs(self, DBU):
        
        '''
        Given a DBU self and another external DBU, present the intersection of 
        one with the other.
        
        intersect_DBU := self.DBU \cap DBU
        
        Parameters:
        -----------------------------------------------------------------------    
        DBU : DisjointBoxUnion
              The DBU we wish to intersect
        
        Returns:
        -----------------------------------------------------------------------
        intersected_DBU : DBU \cap self.DBU
                          The intersected DBU
        
        '''
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for i in range(self.no_of_boxes):
            
            DBU_single = DisjointBoxUnion(1, self.dimension,
                                          np.array([self.lengths[i]]),
                                          np.array([self.centres[i]]),
                                          self.stepsizes[i])
            
            for j in range(DBU.no_of_boxes):
                DBU_2 = DisjointBoxUnion(1, self.dimension,
                                         np.array([DBU.lengths[j]]),
                                         np.array([DBU.centres[j]]),
                                         DBU.stepsizes[j])
                
                #print(f'We remove {DBU_2} from {subtracted_DBU}')
                intersected_DBU = DBU_single.intersect_single_DBUs(DBU_2)
                if intersected_DBU.no_of_boxes != 0:
                    final_DBU = final_DBU.append_disjoint_DBUs(intersected_DBU)
            
        return final_DBU
        
    
    def intersect_single_DBUs(self, DBU):
        
        '''
        Given a DBU with one box and another DBU with just one box, intersect
        the self DBU and the other DBU. The stepsizes are borrowed from the DBU
        we intersect from.
        
        We do not return degenerate rectangles, which are those rectangles which have
        one dimension of the side length be 0
        
        Return self.DBU \cap DBU
        A \cap B = \Pi_{i=1}^D [a_i, b_i] \cap [c_i, d_i]

        where we have A := \Pi_{i=1}^D [a_i, b_i] and B := \Pi_{i=1}^D [c_i, d_i]

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to intersect with the new DBU
                                                                                  
        Returns:
        -----------------------------------------------------------------------
        intersected_DBU : DisjointBoxUnion
                          The DBU we get after intersection     
        '''
        centre1 = self.centres
        lengths1 = self.lengths
        centre2 = DBU.centres
        lengths2 = DBU.lengths
        stepsizes = self.stepsizes
        
        #print(f'We subtract C={centre1[0]} and L={lengths1[0]} and C={centre2[0]} and L = {lengths2[0]}')
        
        rectangle_list = op.intersect_rectangles(centre1[0], lengths1[0],
                                                 centre2[0], lengths2[0])
        
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for rect_no in range(len(rectangle_list)):
            
            centres = np.expand_dims(rectangle_list[rect_no][0], axis=0)
            non_lengths = rectangle_list[rect_no][1]
            
            if not DisjointBoxUnion.has_zero(non_lengths):
                lengths = np.expand_dims(non_lengths, axis=0)
                
                final_DBU= final_DBU.append_disjoint_DBUs(DisjointBoxUnion(1,
                                                                           self.dimension,
                                                                           lengths,
                                                                           centres,
                                                                           stepsizes))
            
        return final_DBU
        
    
    def subtract_DBUs(self, DBU):
        '''
        Given a DBU self and another DBU, subtract one from the other.
        subracted_DBU := self.DBU - DBU
        
        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU
        
        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction
        '''
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for i in range(self.no_of_boxes):
            
            DBU_single = DisjointBoxUnion(1, self.dimension,
                                          np.array([self.lengths[i]]),
                                          np.array([self.centres[i]]),
                                          self.stepsizes[i])
            
            subtracted_DBU = DBU_single
            #print(f'DBU single is {DBU_single}')
            for j in range(DBU.no_of_boxes):
                DBU_2 = DisjointBoxUnion(1, self.dimension,
                                         np.array([DBU.lengths[j]]),
                                         np.array([DBU.centres[j]]),
                                         DBU.stepsizes[j])
                
                #print(f'We remove {DBU_2} from {subtracted_DBU}')
                subtracted_DBU = subtracted_DBU.subtract_single_DBUs(DBU_2)
                if subtracted_DBU.no_of_boxes == 0:
                    break

            if subtracted_DBU.no_of_boxes != 0:    
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)
            
        return final_DBU
    
    @staticmethod
    def has_zero(arr):
        return np.any(arr == 0)
    
    
    def subtract_single_DBUs(self, DBU):
        '''
        Given a DBU with one box and another DBU with just one box, subtract
        the self DBU from the other DBU. The stepsizes are borrowed from the DBU
        we subtract from.
        
        We do not return degenerate rectangles, which are those rectangles which have
        one dimension of the side length be 0
        
        Return self.DBU - DBU

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU.
                                                                                  
        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction       
        '''
        centre1 = self.centres
        lengths1 = self.lengths
        centre2 = DBU.centres
        lengths2 = DBU.lengths
        stepsizes = self.stepsizes
        
        #print(f'We subtract C={centre1[0]} and L={lengths1[0]} and C={centre2[0]} and L = {lengths2[0]}')
        
        rectangle_list = op.subtract_rectangles(centre1[0], lengths1[0],
                                                centre2[0], lengths2[0])
        
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for rect_no in range(len(rectangle_list)):
            
            centres = np.expand_dims(rectangle_list[rect_no][0], axis=0)
            non_lengths = rectangle_list[rect_no][1]
            
            if not DisjointBoxUnion.has_zero(non_lengths):
                lengths = np.expand_dims(non_lengths, axis=0)
                
                final_DBU= final_DBU.append_disjoint_DBUs(DisjointBoxUnion(1,
                                                                       self.dimension,
                                                                       lengths,
                                                                       centres,
                                                                       stepsizes))
            
        return final_DBU
    
    def append_disjoint_DBUs(self, DBU):
        '''
        Append a DBU to the original DBU

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The new DBU to append to the original DBU

        Returns:
        -----------------------------------------------------------------------
        appended_DBU : DisjointBoxUnion
                       The merged DBU

        '''
        new_box_number = DBU.no_of_boxes + self.no_of_boxes
        
        if np.sum(self.lengths) == 0:
            new_lengths = DBU.lengths
            new_centres = DBU.centres
            new_stepsizes = DBU.stepsizes
        else:
            new_lengths = np.concatenate((self.lengths, DBU.lengths), axis = 0)
            new_centres = np.concatenate((self.centres, DBU.centres), axis = 0)
            new_stepsizes = np.concatenate((self.stepsizes, DBU.stepsizes), axis =0)
        
        return DisjointBoxUnion(new_box_number, self.dimension, new_lengths, new_centres,
                                new_stepsizes)
    
    def DBU_union(self, DBU):
        '''
        Given a self DBU, take the union of this with the external DBU.
        
        # A1 sqcup A2 sqcup ... Ak union B = B sqcup A-B
        
        Returns:
        -----------------------------------------------------------------------
            new_DBU : DisjointBoxUnion
                      The new data structure which contains the union with the 
                      old DBU
        '''
        
        
        sub_DBU = self.subtract_DBUs(DBU)
        
        return DBU.append_disjoint_DBUs(sub_DBU)
        
    
    def get_bounds(self):
        '''
        Given a DBU get bounds on the various boxes present in the disjoint union.
        
        Returns:
        -----------------------------------------------------------------------
            bounds : List(2 * d) of length N
                     The bounds on the various boxes present in the DBU
        '''
        bounds = []
        
        for box_no in range(self.no_of_boxes):
            bounds.append([])
            for d in range(self.dimension):
    
                c = self.centres[box_no,d]
                l = self.lengths[box_no,d]
                
                l_bound = c - l/2    
                r_bound = c + l/2

                bounds[-1].append([l_bound, r_bound])
                
        self.bounds = bounds
        
        return bounds
    
    def sample_from_dbu(self):
        '''
        Given a DisjointBoxUnion, sample a point randomly from it.
        
        '''
        box_no = random.randrange(self.no_of_boxes)
        lengths = [DBUIterator.count_points(
                   self.bounds[box_no][d][0],
                   self.bounds[box_no][d][1]-1e-12,
                   self.stepsizes[box_no, d]) for
                   d in range(self.dimension)]
        
        point_index = [random.randrange(l) for l in lengths]
        sampled_point = DBUIterator.index_to_point(point_index, self, box_no)
        
        return sampled_point
    
    
    def sample_from_box(self, box_no):
        '''
        Given a DisjointBoxUnion, sample from a certain box with a given box_no.

        Parameters:
        -----------------------------------------------------------------------
        box_no : int
                 The box number from the DBU we wish to sample from.

        Returns:
        -----------------------------------------------------------------------
        point : np.array[d]
                The new point obtained after sampling
        '''
        lengths = [DBUIterator.count_points(
                   self.bounds[box_no][d][0],
                   self.bounds[box_no][d][1] - 1e-12,
                   self.stepsizes[box_no, d]) for
                   d in range(self.dimension)]
        
        
        point_index = [random.randrange(l) for l in lengths]
        sampled_point = DBUIterator.index_to_point(point_index, self, box_no)
        
        return sampled_point
    
    
    @staticmethod
    def condition_to_DBU(condition, stepsizes):
        '''
        Given a constraint condition, create the corresponding DBU with one box
                                                                                
        Parameters:
        -----------------------------------------------------------------------
        condition : type(constraint_conditions)
            	    The condition which we want to get a DBU from
        	
        stepsizes : np.array[dimension] or int or float
                    The stepsizes of the new DBU in the different dimensions

        Returns:
        -----------------------------------------------------------------------
        constraint_DBU : DisjointBoxUnion
                         The DBU corresponding to the constraint 
        '''
        centres = []
        lengths = []
        
        if len(condition.state_bounds) > 0:
            
            for d in range(condition.dimension):
                centres.append((condition.state_bounds[d,0] + condition.state_bounds[d,1])/2)
                lengths.append(condition.state_bounds[d,1] - condition.state_bounds[d,0])
        
        else:
            centres = np.zeros(condition.dimension)
            lengths = np.array([np.inf for i in range(condition.dimension)])
        
        centres = np.expand_dims(np.array(centres), axis = 0)
        lengths = np.expand_dims(np.array(lengths), axis = 0)
        
        #print('Non zero indices are')
        #print(condition.non_zero_indices)
        if (condition.bounds.shape == (condition.dimension, condition.dimension)):
            for i,ind in enumerate(condition.non_zero_indices):                           
                
                centres[0, ind] = (condition.bounds[ind, 0] + condition.bounds[ind, 1])/2
                lengths[0, ind] = (condition.bounds[ind,1] - condition.bounds[ind,0])             
                
                if lengths[0, ind] == 0:                                                    
                    return DisjointBoxUnion.empty_DBU(condition.dimension)                  
        
        else:
        
            for i,ind in enumerate(condition.non_zero_indices):                           
                
                centres[0, ind] = (condition.bounds[i, 0] + condition.bounds[i, 1])/2
                lengths[0, ind] = (condition.bounds[i,1] - condition.bounds[i,0])             
                
                if lengths[0, ind] == 0:                                                    
                    return DisjointBoxUnion.empty_DBU(condition.dimension)                                     
                                                                                       
            if type(stepsizes) == int or type(stepsizes) == float:                      
                stepsizes = stepsizes * np.ones((1, condition.dimension))                 
            elif len(stepsizes.shape) == 1:
                stepsizes = np.expand_dims(stepsizes, axis=0)
        
        
        return DisjointBoxUnion(1, condition.dimension, lengths, centres,
                                stepsizes, complexity = condition.complexity)
    
    
    def pick_random_point(self, random_seed=42):
        '''
        Given a DisjointBoxUnion, pick a point randomly from it
        '''
        all_points = []
        if len(self.all_points) == 0:
            dbu_iter_class = DBUIterator(self)
            dbu_iterator = iter(dbu_iter_class)
            for s in dbu_iterator:
                all_points.append(s)
            
            self.all_points = all_points
        
        # Create a local random generator with the provided seed
        rng = random.Random(random_seed)
        return rng.sample(self.all_points, 1)
        
    
    def get_total_bounds(self):
        '''
        Given a DBU get the maximum and the minimum bounds in the various dimensions for the distinct 
        boxes
        
        Returns:
        -----------------------------------------------------------------------
            total_bounds : np.array((dimension * 2))
                           The total bounds for the DBU
        Stores:
        -----------------------------------------------------------------------
            total_bounds : Same description as above
        '''
        total_bounds = np.array([[+np.inf, -np.inf] for d in range(self.dimension)])
        bounds = np.array(self.get_bounds())
        
        for d in range(self.dimension):
            for box_no in range(len(bounds)):
                
                if bounds[box_no,d,0] < total_bounds[d,0]:
                    total_bounds[d, 0] = bounds[box_no, d, 0]
                
                if bounds[box_no,d,1] > total_bounds[d,1]:
                    total_bounds[d,1] = bounds[box_no, d,1]
        
        self.total_bounds = total_bounds
        return total_bounds
    
    
    def get_end_points(self):
        '''
        Given a DBU, obtain the end points of the different boxes.

        Returns:
        -----------------------------------------------------------------------
            end_points : np.array(N * 2^D * D)
                         The end points of the n'th box each of which has d coordinates
                                                                                                         
        '''
        end_points = []
        for box_no in range(self.no_of_boxes):
            
            center = self.centres[box_no]
            side_lengths = np.array(self.lengths[box_no])
        
            # Create an array of all possible combinations of -1 and 1 of length d
            offsets = np.array(np.meshgrid(*[[-0.5, 0.5]] * self.dimension)).T.reshape(-1, self.dimension)
        
            # Scale by the side lengths and add the center to get the endpoints
            vertices = center + offsets * side_lengths                         
            
            end_points.append(vertices)
        
        self.end_points = end_points
        
        return end_points
    
    
    def find_point_in_DBU(self, point):
        '''
        Given a point, find which point it corresponds to in the discretized 
        state space                                                                                                   
        
        Parameters:
        -----------------------------------------------------------------------
        point : np.array[dimension]
                A point which we check whether it belongs to the DBU and
                so which is its closest point.
        '''
        
        bounds = self.get_bounds()
        point_box_no = np.inf
        
        for box_no in range(self.no_of_boxes):
            point_in_box = True                                                 
            for d in range(self.dimension):                                         
                if point[d] >= bounds[box_no][d][1] or point[d] < bounds[box_no][d][0]:
                    point_in_box = False                                        
                    break                                                       
                                                                                        
            if point_in_box:                                                    
                point_box_no = box_no                                          
                break                                                          
                                                                                    
        if point_box_no == np.inf:                                              
            
            closest_index, min_distance, closest_point_on_box = DisjointBoxUnion.find_closest_box_to_point(point, self.centres, self.lengths)
            point_box_no = closest_index                                       
            point = closest_point_on_box                                        
            #print(f'The point on the box is {point}')                           
            #print(f'Point is in box {point_box_no}')                            
        
        closest_point = []
        for i in range(self.dimension):
            
            min_bound = self.centres[point_box_no, i] - self.lengths[point_box_no, i] / 2
            max_bound = self.centres[point_box_no, i] + self.lengths[point_box_no, i] / 2 - self.stepsizes[point_box_no, i]
                                                                                
            closest_coord = round((point[i] - min_bound) / self.stepsizes[point_box_no, i]) * self.stepsizes[point_box_no, i] + min_bound
            closest_coord = max(min_bound, min(closest_coord, max_bound))
            closest_point.append(closest_coord)
    
        return np.array(closest_point)
        
    
    def is_point_in_DBU(self, point):
        '''
        Given a point check if it is in the DBU.
        
        Parameters:
        -----------------------------------------------------------------------
        point : np.array[dimension]
                Check if the given point is in the DBU.
        '''
        bounds = self.get_bounds()
        point_box_no = np.inf
        
        for box_no in range(self.no_of_boxes):
            point_in_box = True                                                 
            for d in range(self.dimension):                                         
                if point[d] >= bounds[box_no][d][1] or point[d] < bounds[box_no][d][0]:
                    point_in_box = False                                        
                    break                                                       
                                                                                        
            if point_in_box:                                                    
                point_box_no = box_no
                return True                                          
        
        
        if point_box_no == np.inf:
            return False
    
    @staticmethod
    def integrate_static(dbu, function,
                         check_positivity = False, function_name=''):
        
        '''
        Integrate a function over a dbu, while optionally checking whether f is positive.

        Parameters:
        -----------------------------------------------------------------------
        dbu : DisjointBoxUnion
              The DBU we integrate over
              
        function : func                                                        
                   The function that we integrate with                         

        Returns:
        -----------------------------------------------------------------------
        integral : float
                   The total integral of the function over the dbu
        '''
        
        if dbu.bounds == None:
            dbu.get_bounds()
    
        f_sum = 0.0
        
        for box_no in range(dbu.no_of_boxes):
            coord_list = []
            for dim in range(dbu.dimension):
                
                coord_list.append(np.arange(dbu.bounds[box_no][dim][0],
                                            dbu.bounds[box_no][dim][1] - 1e-12,
                                            dbu.stepsizes[box_no, dim]))         
        
            for point in itertools.product(*coord_list):
                
                y_val = function(np.array(point))
                
                if check_positivity == True:
                    if y_val < 0:
                        raise ValueError(f"Function {function_name} is negative at point {point}: {y_val}")
                    
                f_sum = function(np.array(point)) + f_sum
            
        return f_sum
    
    
    @staticmethod
    def trajectory_integrate(states, f, cond_DBU, t):
        '''
        Perform the integral of function over the trajectory measure.

        Parameters:
        -----------------------------------------------------------------------
        states : list[list]
                 The trajectory measure we wish to integrate over.
                 
        f : func
            A function defined over S the state space
                   
                   
        cond_DBU : DisjointBoxUnion
                   The conditional DBU over which we check existence of the trajectory
                   measure
        
        t    : int
               The time point over the state space we wish to integrate over

        Returns:
        -----------------------------------------------------------------------
        trajectory_integral : float
                              1/len(states) * \sum_{i=1}^{len(states)} f(X_{it},a) I[X_{it} \in cond_DBU]

        '''
        
        trajectory_integral = 0
        for i in range(len(states)):
            point_in_DBU = cond_DBU.is_point_in_DBU(states[i][t])
            if point_in_DBU:
                trajectory_integral += f(states[i][t])
                                                                                
        trajectory_integral = trajectory_integral
        
        return trajectory_integral
    
    
    @staticmethod
    def easy_integral(dbu, function):
        
        '''
        Perform an easy integral of the function over the dbu. This involves
        taking the average of the function over the different boxes
        
        Parameters:
        -----------------------------------------------------------------------
        dbu : DisjointBoxUnion
              The DBU we average over
        
        function : func
                   The function that we integrate with
        
        Returns:
        -----------------------------------------------------------------------
        integral : float
                   The integral value we obtain
        '''
        integral_sum = 0
        endpoints = np.array(dbu.get_end_points())
        for i in range(dbu.no_of_boxes):
            box_sum = 0
            for c in range(endpoints.shape[1]):
                box_sum += function(endpoints[i,c,:])
            
            
            integral_sum += box_sum/ 2**(dbu.dimension)
        
        return integral_sum 
    
    
    @staticmethod
    def n_quad_integrate(dbu, function):
        '''
        Perform an n_quad integral of the function over the dbu. 
        
        Parameters:
        -----------------------------------------------------------------------
        dbu : DisjointBoxUnion
              The DBU we average over
        
        function : func
                   The function that we integrate with
        
        Returns:
        -----------------------------------------------------------------------
        integral : float
                   The integral value we obtain
        '''
        integral_sum = 0
        bounds = np.array(dbu.get_bounds())
        for i in range(dbu.no_of_boxes):
            print(f'{i}th bounds is')
            print(bounds[i])
            integral_sum += integrate.nquad(function, bounds[i])
        
        return integral_sum / dbu.no_of_boxes
    
    @staticmethod
    def sampling_integrate_static(dbu, function,
                                  point_percentages = 0.3):
        '''
        Integrate a function over a disjoint box union data structure. We assume
        to do a MCMC based sampling integral by sampling points_per_box number of points
        or point_percentage number of points in each box.
        
        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d dimensional input function to integrate over the DBU.
        
            
            point_percentages : float or list
                               The percentage of points we sample from each 
                               box of the DBU to perform the integration of the function
        
        Returns:
        -----------------------------------------------------------------------
            integral : float
                       The estimate of the Riemannian summation integral over the DBU. 
        
        '''
        
        if type(point_percentages) == float:
            point_percentages = [point_percentages for i in range(dbu.no_of_boxes)]
        
        
        sampled_points = 0
        function_sum = 0
        for box_no in range(dbu.no_of_boxes):
            
            lengths = [DBUIterator.count_points(
                                   dbu.bounds[box_no][d][0],
                                   dbu.bounds[box_no][d][1] - 1e-12,
                                   dbu.stepsizes[box_no, d]) for
                                   d in range(dbu.dimension)]
            
            points_per_box = 1
            for l in lengths:
                points_per_box = points_per_box * l
        
            sampling_number = int(points_per_box * point_percentages[box_no])
            if sampling_number == 0:
                sampling_number = 1
            sampled_points += sampling_number
            
            for s in range(sampling_number):
                
                function_sum += function(dbu.sample_from_box(box_no))
        
        if sampled_points == 0:
            sampled_points = 1
        #print(f'Sampled points is {sampled_points}')
        return function_sum
    
    
    @staticmethod
    def find_closest_box_to_point(P, box_centers, box_lengths):
        P = np.array(P)
        closest_index = None
        min_distance = float('inf')
        closest_point_on_box = None
        
        for i, (center, length) in enumerate(zip(box_centers, box_lengths)):
            center = np.array(center)
            length = np.array(length) / 2  # Half-lengths
            
            # Compute the minimum distance to the box edges along each dimension
            lower_bounds = center - length
            upper_bounds = center + length
            
            # Distance to the nearest point on the box surface
            clamped_point = np.maximum(lower_bounds, np.minimum(P, upper_bounds))
            distance = np.linalg.norm(P - clamped_point)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                closest_point_on_box = clamped_point
        
        return closest_index, min_distance, closest_point_on_box
    
    
    @staticmethod
    def generate_k_tuples(array_list, k):                                                                   
        '''
        Given a list of lists, return all possible k-tuples from the list of lists,
        and also return the indices. 

        Parameters:
        -----------------------------------------------------------------------
        lists : list[list]
                The list of lists we wish to sample from
        k : int
            The number of elements we want to look at in the tuples

        Returns:
        -----------------------------------------------------------------------
        results : tuple
                  The tuple which denotes the values and the indices we wish to
                  sample from 

        '''
        # Step 3: Generate all possible combinations of indices of size k
        index_combinations = list(itertools.combinations(range(len(array_list)), k))

        # Step 4: Generate all possible k-tuples for each combination
        all_k_tuples = []
        for indices in index_combinations:
            # Extract the sub-arrays corresponding to the current combination of indices
            sub_arrays = []
            for i in indices:
                cart_arr = itertools.product(array_list[i], repeat = 2)
                unequal_tuples = [tup for tup in cart_arr if tup[0]<tup[1]]
                sub_arrays.append(unequal_tuples)

            # Generate all possible k-tuples from the sub-arrays
            k_tuples = list(itertools.product(*sub_arrays))
            # Store the results
            all_k_tuples.append((indices, k_tuples))

    
        return all_k_tuples
    
    @staticmethod
    def tuples_to_array(tup):
        '''
        Given a tuple of tuples convert it into an array
        '''
        final_arr = []
        for i,elt in enumerate(tup):
            final_arr.append(np.array(elt))
        
        return np.array(final_arr)
    
    @staticmethod
    def dp_integrate(dbu, function):
        '''
        Integrate a function over the DBU using dynamic programming 
        to avoid redundant computations.
        
        Parameters:
            -----------------------------------------------------------------------
            function : func
                       The d-dimensional function to integrate over the DBU
                     
        Returns:         
            -----------------------------------------------------------------------
        integral : float
                   The value of the integral of the function over the DBU
                   
        '''
    
        # Dictionary to store computed integrals for sub-regions to avoid recomputation
        dp_cache = {}
        
        total_integral = 0.0
        
        for box_no in range(dbu.no_of_boxes):
            box_integral = 0.0
            coord_list = []
        
            # Create coordinate ranges for each dimension in the current box
            for dim in range(dbu.dimension):
                coord_list.append(np.arange(
                    dbu.bounds[box_no][dim][0],
                    dbu.bounds[box_no][dim][1] -1e-12 ,
                    dbu.stepsizes[box_no, dim]
                    ))
        
            # Iterate over all points (sub-regions) in the grid of the current box
            for point in itertools.product(*coord_list):
                point_key = tuple(point)
            
            # Check if this sub-region integral is already cached
            if point_key not in dp_cache:
                # If not, compute the function at this point and store it
                dp_cache[point_key] = function(np.array(point))
            
                # Add the cached value to the box integral
                box_integral += dp_cache[point_key]
        
            # Add the box integral to the total integral
            total_integral += box_integral
    
        # Normalize by the total number of points in the DBU
        return total_integral / dbu.no_of_points()
    
    
    #Create an iterator method for conditions to speed up generate all conditions?
    
    def generate_all_conditions(self, max_complexity):
        '''
        For the given DBU, generate all the bounded conditions that is possible for
        a given complexity.
        
        Parameters:
        -----------------------------------------------------------------------
            max_complexity : int
                             Possible number of coordinates that can be modified in
                             the DBU.
            
        
        Returns:
        -----------------------------------------------------------------------
            conditions : list[constraint_conditions]
                         The list of bounded conditions that can be generated for a given
                         complexity for the given DBU.
        '''
        
        condition_list = []
        total_bounds = self.get_total_bounds()
        
        for box_no in range(self.no_of_boxes):
            cartesian_prod = []
            for dim in range(self.dimension):
                
                cart = np.arange(self.get_bounds()[box_no][dim][0],
                                 self.get_bounds()[box_no][dim][1] -1e-12,
                                 self.stepsizes[box_no, dim])
                
                cartesian_prod.append(cart)
                
            for k in range(1, max_complexity+1):
                all_k_tuples = DisjointBoxUnion.generate_k_tuples(cartesian_prod, k)
                
                for indices, k_tuples in all_k_tuples:
                    for k_tuple in k_tuples:
                        
                        k_array = DisjointBoxUnion.tuples_to_array(k_tuple)
                        
                        constraint = cc.ConstraintConditions(dimension = self.dimension,
                                                             non_zero_indices = np.array(indices),
                                                             bounds = k_array,
                                                             state_bounds = total_bounds)
                        
                        condition_list.append(constraint)
            
        return condition_list
    
    
    def generate_conditions_from_observations(self, obs_states, max_complexity):
        
        """
        Generate condition boxes (axis-aligned intervals) using observed states.
        For each subset of dimensions (up to max_complexity), iterate over all 
        lower and upper value combinations to form intervals.
        
        Parameters :
        -----------------------------------------------------------------------
        obs_states : list[states]
                     Observed states at each time and lengthstep
                     
     max_complexity : int
                      The maximal complexity for each condition
        
        Returns :
        -----------------------------------------------------------------------
        condition_list : list[constraint_condition]
                         The conditions generate with the different upper and lower bounds 
        
        """
        
        all_points = []
        for t_states in obs_states:
            for state in t_states:
                all_points.append(np.array(state))
    
        all_points = np.array(all_points)  # shape: (n, d)
        dimension = self.dimension
    
        # Collect unique sorted values for each dimension
        dim_values = {d: sorted(set(all_points[:, d])) for d in range(dimension)}
    
        condition_list = []
        state_bounds = self.get_total_bounds()
    
        for k in range(1, max_complexity + 1):
            for dims in combinations(range(dimension), k):
                value_lists = [dim_values[d] for d in dims]
    
                # Iterate over all combinations of lower bounds
                for lower_combo in product(*value_lists):
                    # For each lower bound, consider upper bounds >= that value in the same dim
                    upper_lists = []
                    valid = True
                    for i, d in enumerate(dims):
                        vlist = [v for v in dim_values[d] if v >= lower_combo[i]]
                        if not vlist:
                            valid = False
                            break
                        upper_lists.append(vlist)
    
                    if not valid:
                        continue
    
                    for upper_combo in product(*upper_lists):
                        # Require lower <= upper component-wise
                        if any(l > u for l, u in zip(lower_combo, upper_combo)):
                            continue
    
                        total_bounds = np.array([[l, u] for l, u in zip(lower_combo, upper_combo)])
                        
                        constraint = cc.ConstraintConditions(
                                        dimension=dimension,
                                        non_zero_indices=np.array(dims),
                                        bounds=total_bounds,
                                        state_bounds=state_bounds
                                        )   
                        
                        condition_list.append(constraint)
    
        return condition_list
    
    
        
    def plot_2D(self, ax=None, show_plot=True, title=''):
        '''
        Plot a disjoint union of boxes in 2D
        '''
        if ax == None:
            fig,ax = plt.subplots()
        
        if self.bounds == None:
            self.get_bounds()
        #print(self.total_bounds)
        #print(type(self.total_bounds))
        
        for box_no in range(self.no_of_boxes):
            ax.scatter([self.centres[box_no, 0]], [self.centres[box_no, 1]],
                       color = 'red', marker = 'x')
            
            ax.hlines(y=self.centres[box_no, 1] - self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2)
            
            ax.hlines(y=self.centres[box_no, 1] + self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2,
                       linestyle = 'dotted')
            
            ax.vlines(x=self.centres[box_no, 0] + self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2,
                       linestyle = 'dotted')
            
            ax.vlines(x=self.centres[box_no, 0] - self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2)
        
        #ax.set_xlim(self.total_bounds[0,0]-1, self.total_bounds[1,0]+1)
        #ax.set_ylim(self.total_bounds[0,1]-1, self.total_bounds[1,1]+1)
        ax.set_title(title)
        if show_plot:
            plt.show()
            
        return ax
    
    
    def __str__(self):
        '''
        Print the various rectangles present in the DBU.
        
        '''
        dbu_string = ''
        for box_no in range(self.no_of_boxes):
            dbu_string += f'Box: {box_no}, Centre: {self.centres[box_no]}, Lengths: {self.lengths[box_no]}, Stepsizes: {self.stepsizes[box_no]}\n'
        
        print(f'Number of boxes is {self.no_of_boxes}')
        return dbu_string    
    
    def dbu_to_tuple(self):
        '''
        Given a DBU get the tuple associated to it which is to be
        used in storing the dicts
        '''
        return (self.no_of_boxes, self.dimension,
                DisjointBoxUnion.tuplify_2D_array(self.lengths),
                DisjointBoxUnion.tuplify_2D_array(self.centres))
    
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
        
    
    @staticmethod                                                              
    def empty_DBU(dimension):                                                  
        '''
        A static method that returns an empty DBU of a certain dimension     
        Parameters:                                                             
        -----------------------------------------------------------------------
        dimension : int
                    Dimension of the empty DBU

        Returns:
        -----------------------------------------------------------------------
        empty_DBU : type(DisjointBoxUnion)
                    An empty disjoint box union data structure

        '''
        return DisjointBoxUnion(0, dimension, np.array([[]]), np.array([[]]))

#%%
    
class DBUIterator:
    '''
    Create a DBU iterator for iterating through the points of the different boxes
    '''
    def __init__(self, DBU):
    
        self.DBU = DBU
        self.curr_box_no = 0
        self.point_index = np.zeros(self.DBU.dimension)
        self.first_point = True
        
    @staticmethod
    def count_points(a, b, d):
        '''
        Count the number of points in the list given by np.arange(a,b,d)

        Parameters:
        -----------------------------------------------------------------------
        a : float
            Lower Bound
        b : float
            Upper Bound
        d : float
            Difference

        Returns:
        -----------------------------------------------------------------------
        number_of_points : int
                           The number of points in the list

        '''
        return max(0, int(np.ceil(b-a) / d))
    
    @staticmethod
    def index_to_point(point_index, DBU, box_no):
        '''
        Given the set of indices for a certain box_no for a DBU return the corr
        point on the DBU

        Parameters:
        -----------------------------------------------------------------------
        point_index : np.array([int])
                      The indices of the point in the box

        Returns:
        -----------------------------------------------------------------------
        point : np.array[float]
                The point values on the box
        '''
        point = []
        for d in range(DBU.dimension):
            
            point.append(DBU.bounds[box_no][d][0] + (point_index[d]) * DBU.stepsizes[box_no, d])
            
        return point
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            self.lengths = [DBUIterator.count_points(
                                   self.DBU.bounds[box_no][d][0],
                                   self.DBU.bounds[box_no][d][1],
                                   self.DBU.stepsizes[box_no, d]) for
                                   d in range(self.DBU.dimension)]
            
            if self.first_point:
                self.first_point = False
                return DBUIterator.index_to_point(np.zeros(self.DBU.dimension), self.DBU, box_no)
                
            
            # Update the position for the next point
            for i in range(self.DBU.dimension - 1, -1, -1):

                if self.point_index[i] + 1 < self.lengths[i]:
                    self.point_index[i] += 1
                    point = DBUIterator.index_to_point(self.point_index ,self.DBU, box_no)
                    # We manually add this change so that we can return the true point
                    
                    return point
                else:
                    self.point_index[i] = 0
                    #print(f'Point index : {point_index}')
                    #print(f'Else condition reached and i is {i}')
                    if i == 0:
                        self.curr_box_no += 1
                        self.first_point = True
                
            self.point_index = np.zeros(self.DBU.dimension)
            
        if self.curr_box_no >= self.DBU.no_of_boxes:
            raise StopIteration

#%%
class ConditionsIterator:
    '''
    For a DBU and a given box we define an iterator to iterate through all the conditions
    that could possibly be generated with this method.
    '''
    
    def __init__(self, DBU, max_complexity, cond_stepsizes = 0.0):
        '''
        We generate and iterate through all the possible conditions

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU over which we wish to generate all possible conditions
              
        curr_box_no : int
                 The box number in the DBU for which we wish to generate all possible constraint conditions
                 
        max_complexity : int
                         The maximum allowed complexity for a constraint condition

        Returns:
        -----------------------------------------------------------------------
        condition_iter class : ConditionIterator
                               The class which generates the iterations over all the conditions

        '''
        
        self.DBU = DBU
        self.max_complexity = max_complexity
        
        self.curr_box_no = 0
        self.complexity = 1
        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
        
        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
        self.bounds_per_box = DBU.get_bounds()
        self.curr_bounds = np.array(DBU.get_bounds()[0])
        self.total_bounds = DBU.get_total_bounds()
        
        if cond_stepsizes == 0.0:
            self.cond_stepsizes = self.DBU.stepsizes
        elif type(cond_stepsizes) == int or type(cond_stepsizes) == float:
            self.cond_stepsizes = np.ones((self.DBU.no_of_boxes, self.DBU.dimension))
        else:
            self.cond_stepsizes = cond_stepsizes
        
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            for complexity in range(self.complexity, self.max_complexity+1):
                
                for non_zero_indices in self.possible_non_zero_indices[-self.curr_non_zero_index_position:]:
                    non_zero_indices = np.array(non_zero_indices)
                    
                    for d in non_zero_indices[-self.iter_dim:]:
                        lengthstep = self.cond_stepsizes[box_no, d]
                        #print(f'We check if {curr_bounds[d,0]} + {lengthstep} < {curr_bounds[d,1]}')
                        if self.curr_bounds[d, 0] + lengthstep < self.curr_bounds[d, 1] - 1e-12:
                            self.curr_bounds[d, 0] += lengthstep
                            #print(f'New left bounds {self.curr_bounds[d,0]}')
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                            constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds,
                                                                 self.total_bounds)
                            #print(f'The constraint is {constraint}')
                            #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                            return constraint
                        
                        elif self.curr_bounds[d, 1] - lengthstep > np.array(self.bounds_per_box[box_no])[d,0]:
                            
                             self.curr_bounds[d, 1] -= lengthstep
                             #print(f'New right bounds {self.curr_bounds[d,1]}')
                             self.curr_bounds[d, 0] = np.array(self.bounds_per_box[box_no])[d, 0]
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                             constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds,
                                                                 self.total_bounds)
                            #print(f'The constraint is {constraint}')
                             #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                             return constraint
                        
                        else:
                            self.curr_non_zero_index_position -= 1
                            
                            if self.curr_non_zero_index_position > 0:
                                self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                            
                            else:
                                self.iter_dim -= 1
                                if self.iter_dim > 0:
                                    self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                    self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                else:
                                    self.complexity += 1
                                    if self.complexity <= self.max_complexity: 
                                        self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                    
                                    else:
                                        self.curr_box_no += 1
                                        if self.curr_box_no < self.DBU.no_of_boxes:
                                            self.complexity = 1
                                            self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                            self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                            self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                            self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                                                                
                                        else:                                   
                                            #print('Hurray! We reached the endpoint')                                    
                                            raise StopIteration                
                            
#%%

if __name__ == '__main__':
    
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    dbu.get_bounds()[1]
    
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    
    constraint_list = dbu.generate_all_conditions(2)
    fig,ax = plt.subplots()
    
    ax = dbu.plot_2D(ax, show_plot = False)
    for i,constraint in enumerate(constraint_list):
        constraint.plot_2D_constraints(ax, show_plot = True)
        fig, ax = plt.subplots()
        ax = dbu.plot_2D(ax, show_plot=False)
    
    for i,constraint in enumerate(constraint_list):
        print(f'The {i}th constraint is {constraint}')                         

    #%%

    dbu1 = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    dbu2 = DisjointBoxUnion(2, 2, np.array([[3,3], [1,1]]), np.array([[0,0], [-3,-3]]))
    
    final_dbu = dbu1.subtract_DBUs(dbu2)
    
    
    print('Final dbu is')
    print(final_dbu)