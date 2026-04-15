# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 21:57:39 2026
@author: ggv16

Module: Synaptic Weight Manager
Description: Handles the persistence layer of the SNN.
             Responsible for serializing (saving) trained synaptic weights and 
             reconstructing (loading) the exact network topology for inference.
"""

import brian2 as b2
import pickle
import os
import numpy as np

def get_weights_dir():
    """
    Resolves the absolute path to the 'saved_weights' directory.
    Ensures cross-platform compatibility and directory existence.
    """
    # Navigate to the project root relative to this script location
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(root_dir, 'saved_weights')
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def save_weights(network_objects, filename='weights.pkl', metadata=None):
    """
    Serializes the state of the trained network to disk.
    separately stores connectivity matrices (topology) and synaptic strengths (weights)
    for both the 'Healthy' and 'Arrhythmia' populations.

    Args:
        network_objects (dict): Dictionary containing the Brian2 Synapse objects.
        filename (str): Target filename for the pickle archive.
        metadata (dict, optional): Additional training context (e.g., epoch count, accuracy).
    """
    filepath = os.path.join(get_weights_dir(), filename)
    
    # Extract Synapse Objects from the network dictionary
    syn_sano = network_objects['synapses_sano']
    syn_arr = network_objects['synapses_arr']
    
    # Structure the data for persistence
    # We store raw Numpy arrays to avoid Brian2 unit serialization issues
    data = {
        'metadata': metadata,
        
        # Team Healthy: "Sharp-shooter" weights
        'sano': {
            'indices_i': np.array(syn_sano.i[:]), # Pre-synaptic indices
            'indices_j': np.array(syn_sano.j[:]), # Post-synaptic indices
            'w': np.array(syn_sano.w[:])          # Learned weights (Voltage)
        },
        
        # Team Arrhythmia: "Integrator" weights
        'arritmia': {
            'indices_i': np.array(syn_arr.i[:]),
            'indices_j': np.array(syn_arr.j[:]),
            'w': np.array(syn_arr.w[:])
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
        
    print(f" Weights successfully serialized to: {filepath}")

def load_weights(network_objects, filename='weights.pkl'):
    """
    Reconstructs the trained network topology and injects synaptic weights.
    
    This function creates the physical connections (synapses) based on the 
    stored indices before assigning the weight values. This is critical for 
    reproducing the exact sparse connectivity pattern learned during training.

    Args:
        network_objects (dict): Dictionary containing the target (empty) Synapse objects.
        filename (str): Name of the source weights file.
    """
    filepath = os.path.join(get_weights_dir(), filename)
    if not os.path.exists(filepath):
        print(f" Error: Weights file not found at {filepath}")
        return

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    syn_sano = network_objects['synapses_sano']
    syn_arr = network_objects['synapses_arr']
    
    # TOPOLOGY RECONSTRUCTION STRATEGY 
    
    # 1. Deactivate Synapses temporarily to prevent simulation artifacts during loading
    syn_sano.active = False 
    syn_arr.active = False
    
    # 2. Re-establish Physical Connections (The "Wiring")
    # The .connect() method creates the synapses between specific neuron pairs (i, j).
    # This restores the structural memory of the network.
    syn_sano.connect(i=data['sano']['indices_i'], j=data['sano']['indices_j'])
    syn_arr.connect(i=data['arritmia']['indices_i'], j=data['arritmia']['indices_j'])
    
    # 3. Inject Synaptic Weights (The "Strength")
    # We must re-apply the physical units (b2.volt) as they were stripped during saving.
    syn_sano.w = data['sano']['w'] * b2.volt 
    syn_arr.w = data['arritmia']['w'] * b2.volt
    
    # 4. Reactivate Synapses for Inference
    syn_sano.active = True
    syn_arr.active = True
    
    print(f" Memory Restored. Healthy Team: {len(syn_sano)} synapses | Arrhythmia Team: {len(syn_arr)} synapses.")

def load_topology(filename='weights.pkl'):
    """
    Helper function to inspect the stored network topology without loading the full model.
    Useful for debugging connectivity patterns or sparse matrix analysis.
    
    Returns:
        dict: Containing (i, j) index tuples for both populations.
    """
    filepath = os.path.join(get_weights_dir(), filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    return {
        'sano': (data['sano']['indices_i'], data['sano']['indices_j']),
        'arritmia': (data['arritmia']['indices_i'], data['arritmia']['indices_j'])
    }