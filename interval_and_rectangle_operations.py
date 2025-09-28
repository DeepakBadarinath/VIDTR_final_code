#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:10:05 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


import icecream
from icecream import ic
#%%

# Library with operations related to intervals - Used as backend for the disjoint_box_union module


def intersect_intervals(centre1, lengths1, centre2, lengths2):
    '''      
    Given two intervals, find their intersection
             
    Parameters:
    -----------------------------------------------------------------------
    centre1 : float
              Centre of the first interval
    lengths1 : float
              Length of the first interval
    centre2 : float
              Centre of the second interval
    lengths2 : float
              Length of the second interval
              
    Returns:
    -----------------------------------------------------------------------
    centre : float
             The centre of the intersection interval
    length : float
             The length of the intersection interval

    '''
    #Disjoint
    if centre1 + lengths1/2 <= centre2 - lengths2/2:
        return (centre1+centre2)/2, 0.0
    
    #Disjoint
    elif centre2 + lengths2/2 <= centre1 - lengths1/2:
        return (centre1 + centre2)/2, 0.0
    
    #Partial intersection
    elif centre1 - lengths1/2 <= centre2 - lengths2/2 and (
            centre1 + lengths1/2 <= centre2 + lengths2/2):
        return (centre2 - lengths2/2 + centre1 + lengths1/2)/2, centre1+lengths1/2 - (centre2 - lengths2/2)

    #Partial intersection
    elif centre2 - lengths2/2 <= centre1 - lengths1/2 and (
            centre2 + lengths2/2 <= centre1 + lengths1/2):
        return (centre1 + centre2)/2 - lengths1/4 + lengths2/4, centre2 + lengths2/2 - (centre1 - lengths1/2)
    
    #Second interval sits inside first interval
    elif centre1 - lengths1/2 <= centre2 - lengths2/2 and (
            centre1 + lengths1/2 >= centre2 + lengths2/2):
        return centre2, lengths2
    
    #First interval sits inside the second interval 
    else:
        return centre1, lengths1 
    
def subtract_intervals(centre1, lengths1, centre2, lengths2):
    '''
    Given the centres and lengths of two intervals, find their difference

    Parameters
    -----------------------------------------------------------------------
    centre1 : float
              Centre of the first interval
    lengths1 : float
               Length of the first interval
    centre2 : float
              Centre of the second interval
    lengths2 : float
               Length of the second interval

    Returns
    -----------------------------------------------------------------------
    difference_intervals : List[tuple]
                           The centres and lengths of the subtracted intervals
    '''
    
    #Disjoint
    if centre1 + lengths1/2 <= centre2 - lengths2/2:
        new_centre = centre1
        new_lengths = lengths1
        return [(new_centre, new_lengths)]
    
    #Disjoint
    elif centre2 + lengths2/2 <= centre1 - lengths1/2:
        new_centre = centre1
        new_lengths = lengths1
        return [(new_centre, new_lengths)]
        
    
    #Partial intersection
    elif centre1 - lengths1/2 <= centre2 - lengths2/2 and (centre2 + lengths2/2 >= centre1 + lengths1/2):
        new_centre = (centre1 + centre2)/2 - lengths1/4 - lengths2/4
        new_lengths = centre2 - centre1 - lengths2/2 + lengths1/2
        return [(new_centre, new_lengths)]
     
    #Partial intersection
    elif centre2 - lengths2/2 <= centre1 - lengths1/2 and (centre1 + lengths1/2 >= centre2 + lengths2/2):
        
        new_centre = (centre1 + centre2)/2 + (lengths1 + lengths2)/4
        new_lengths = centre1 - centre2 + lengths1/2 - lengths2/2
        return [(new_centre, new_lengths)]
    
    #Second interval sits inside the first interval
    elif centre1 - lengths1/2 <= centre2 - lengths2/2 and (
            centre1 + lengths1/2 > centre2 + lengths2/2):
        
        newcentre1 = (centre1 + centre2)/2 - (lengths1 + lengths2)/4
        newlengths1 = centre2 - lengths2/2 - centre1 + lengths1/2
        
        newcentre2 = 0.5 * (centre1 + lengths1/2 + centre2 + lengths2/2)
        newlengths2 = centre1 + lengths1/2 - centre2 - lengths2/2
        
        return [(newcentre1, newlengths1), (newcentre2, newlengths2)]
    
    
    #First interval sits inside the second interval 
    else:
        
        return [(centre1 + lengths1/2, 0.0)]
    

def union_of_intervals(centres, lengths, additional_centre, additional_length,
                       glue=False):
    '''
    Given a list of disjoint interval centres and lengths and the centre and length of a
    new interval, find the union of the new interval with the given sequence
    
    A_1 sqcup A_2 sqcup A_3 sqcup ... A_k union B = B sqcup sqcup_{i=1}^k (A_i - B)
    
    Parameters:
    -----------------------------------------------------------------------
    centres : np.array                                                     
              The one-dimensional centres of the disjoint intervals        
              
    lengths : np.array                                                     
              The one-dimensional lengths of the disjoint intervals        
    
    additional_centre : float                                              
                 The centre of the interval we wish to unionize            
     
    additional_length : float                                              
                 The length of the interval we wish to unionize            
    
    Returns:                                                               
    -----------------------------------------------------------------------
    new_centres : np.array                                                  
                  The centres after taking the union                       
    
    new_lengths : np.array                                                 
                  The lengths after taking the union                       
    '''
    
    new_centres = []
    new_lengths = []
    
    for i in range(len(centres)):
        sub_intervals = subtract_intervals(centres[i],
                                           lengths[i],
                                           additional_centre,
                                           additional_length)
        
        for no in range(len(sub_intervals)):
            
            if sub_intervals[no][1] > 0:
                new_centres.append(sub_intervals[no][0])
                new_lengths.append(sub_intervals[no][1])
        
    
    new_centres.append(additional_centre)
    new_lengths.append(additional_length)
    
    if glue:
        new_centres, new_lengths = glue_intervals(new_centres, new_lengths)
    
    return np.array(new_centres), np.array(new_lengths)


def glue_intervals(centers, lengths):
    """
    Glue overlapping or touching intervals together.

    Parameters:
    ---------------------------------------------------------------------------
    centers : np.array(float) 
              The centers of the intervals.
             
    lengths : np.array(float)
              The lengths of the intervals.

    Returns:
    ---------------------------------------------------------------------------
    glued_centers : np.array(float)
                    A list of glued interval centers.
                    
    glued_lengths : np.array(float)
                    A list of glue interval
    """
    # Sort based on the start points
    iter_centers = np.copy(centers)
    iter_lengths = np.copy(lengths)
    
    glued_centers  = []
    glued_lengths = []
    
    # Combine centers and lengths into a list of tuples (center, length)
    combined = list(zip(iter_centers, iter_lengths))
    
    # Sort the combined list based on the key centers[i] - lengths[i] / 2
    combined.sort(key=lambda x: x[0] - x[1] / 2)
    
    # Unpack the sorted tuples back into centers and lengths
    iter_centers[:], iter_lengths[:] = zip(*combined)
    i = 0
    # Gluing of the intervals
    while i < (len(iter_centers)):
        if i < len(iter_centers) - 1:
            if iter_centers[i] + iter_lengths[i]/2 >= iter_centers[i+1] - iter_lengths[i+1]/2:
                
                new_c = 0.5 * (iter_centers[i] - iter_lengths[i]/2 + iter_centers[i+1] + iter_lengths[i+1]/2)
                new_l = iter_centers[i+1] + iter_lengths[i+1]/2 - iter_centers[i] + iter_lengths[i]/2
            
                iter_centers[i] = new_c
                iter_lengths[i] = new_l
                
                iter_centers = np.delete(iter_centers, i+1)
                iter_lengths = np.delete(iter_lengths, i+1)
                
            else:
                
                glued_centers.append(iter_centers[i])
                glued_lengths.append(iter_lengths[i])
                i = i + 1
        else:
            
            glued_centers.append(iter_centers[i])
            glued_lengths.append(iter_lengths[i])
            i = i + 1
    
    return np.array(glued_centers), np.array(glued_lengths)
               
def convert_to_intervals(centres, lengths):
    '''
    Given a list of interval centres and lengths, convert it to list of tuples 
    with the given endpoints
    '''
    tuple_list = []
    for i in range(len(centres)):
        tuple_list.append((centres[i] - lengths[i]/2, centres[i] + lengths[i]/2))
    
    return tuple_list


def plot_intervals(centers, lengths, title=None):
    """
    Plots intervals on a graph given their centers and lengths.
    
    Parameters:
    ---------------------------------------------------------------------------
    centers : np.array
              The centers of the intervals.
              
    lengths : np.array
              The lengths of the intervals
    """
    if len(centers) != len(lengths):
        raise ValueError("The length of centers and lengths lists must be the same.")
    
    plt.figure(figsize=(10, 5))
    
    for i, (center, length) in enumerate(zip(centers, lengths)):
        start = center - length / 2
        end = center + length / 2
        
        plt.plot([start, end], [i, i], color='blue', marker='|', markersize=15)
        plt.plot(center, i, 'x', color='red')
        
    plt.yticks(range(len(centers)))
    plt.xlabel('Value')
    
    if title != None:
        plt.title(title)
    else:
        plt.title(title)
    
    plt.grid(True)
    plt.show()


def subtract_rectangles(centre1, lengths1, centre2, lengths2):
    '''
    Given two left half open rectangles,
    one having centre1 and lengths1 and the other with centres2 and lengths2
    Find R1 - R2
    
    Works based on the logic that 
    (A1 \times A) - (B1 \times B) = [(A1-B1) \times A] \sqcup [(A1 \cap B1) \times (A-B)]         

    Parameters:
    -----------------------------------------------------------------------
    centre1 : np.array[d]
              The centre of the first rectangle
    lengths1 : np.array[d]
               The lengths of the sides of the first rectangle
    centre2 : np.array[d]
              The centre of the second rectangle
    lengths2 : np.array[d]
               The lengths of the sides of the second rectangle

    Returns:
    -----------------------------------------------------------------------
    difference_rectangles (d): List[tuples]
                               d[i] = (centre of ith rectangle,
                                       length of ith rectangle)
    '''
    if len(centre1.flatten()) == 1:
        
        return subtract_intervals(centre1[0], lengths1[0],
                                  centre2[0], lengths2[0])
    
    final_rectangles = []
    interval_differences = subtract_intervals(centre1[0],
                                              lengths1[0],
                                              centre2[0],
                                              lengths2[0])
    #A1 - B1
    for box_no in range(len(interval_differences)):
        
        centre = interval_differences[box_no][0]
        length = interval_differences[box_no][1]
        
        #(A1 - B1) \times A
        if length > 0:
            centre_left_union = np.array([centre] + (centre1[1:]).tolist())
            lengths_left_union = np.array([length] + (lengths1[1:]).tolist())                                                       
            final_rectangles.append((centre_left_union, lengths_left_union))
    
    #A1 \cap B1                                    
    right_centre, right_lengths = intersect_intervals(centre1[0],
                                                      lengths1[0], 
                                                      centre2[0],
                                                      lengths2[0])
                                                   
    #A - B                                                                                                                         
    
    if right_lengths>0:
        
        right_diff = subtract_rectangles(centre1[1:], lengths1[1:],
                                         centre2[1:], lengths2[1:])
        
        
        #print(f'Right diff is {right_diff}')
        #(A1 \cap B1) \times (A-B)
        if len(right_diff) != 0:
            for i in range(len(right_diff)):
                
                #print(f'Right centre is {[right_centre]}')
                #print(f'Right diff at {i} and 0 is {right_diff[i][0]} with type being {type(right_diff[i][0])}')
                
                
                if type(right_diff[i][0]) == list or type(right_diff[i][0]) == type(np.array([1,3])):
                    new_c = np.array([right_centre] + list(right_diff[i][0]))
                
                else:
                    new_c = np.array([right_centre] + [right_diff[i][0]])
                
                #print(f'New c is {new_c}')
                #print(f'Right lengths are {right_lengths} and right_diff at {i} and 1 is {right_diff[i][1]}')
                
                if type(right_diff[i][1]) == list or type(right_diff[i][1]) == type(np.array([1,2])):
                    new_l = np.array([right_lengths] + list(right_diff[i][1]))
                else:
                    new_l = np.array([right_lengths] + [right_diff[i][1]])

                #print(f'New lengths are {new_l}')
                final_rectangles.append((new_c, new_l))
    
    #print(f'We return the final rectangles : {final_rectangles}')            
    return final_rectangles


#%%