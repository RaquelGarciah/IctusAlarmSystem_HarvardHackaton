# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:15:00 2026
@author: ggv16

Module: Neuromorphic Data Visualization
Description: Generates publication-quality plots for SNN analysis.
             Handles both Brian2 Monitors and lightweight SimResult objects.
"""

import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

def plot_clinical_validation(sim_result, title="Clinical Diagnosis", 
                             threshold_sano=-20, threshold_arr=-30):
    """
    Plots the membrane potential dynamics from a SimResult object.
    Focuses on the competition between the 'Healthy' and 'Arrhythmia' teams.

    Args:
        sim_result: The SimResult object returned by compare_recognition.
        title (str): Plot title (e.g., "Patient 115").
        threshold_sano (float): Threshold for Healthy detection (mV).
        threshold_arr (float): Threshold for Arrhythmia detection (mV).
    """
    
    # 1. DATA EXTRACTION & UNIT STRIPPING 
    
    # Process Time
    if hasattr(sim_result.t, 'dim'): # It's a Brian2 quantity
        times = sim_result.t / b2.ms
    else: # It's a raw numpy array
        times = sim_result.t * 1000 

    # Process Voltage
    if hasattr(sim_result.v, 'dim'):
        v_traces = sim_result.v / b2.mV
    else:
        v_traces = sim_result.v * 1000

    # 2. PLOTTING SETUP
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine the split between teams
    n_neurons = v_traces.shape[0]
    mid_point = n_neurons // 2
    
    # 3. DRAW TRACES
    # Plot Team Healthy (Green)
    label_sano = "Team Healthy (Sync)"
    for i in range(mid_point):
        ax.plot(times, v_traces[i], color='forestgreen', alpha=0.5, linewidth=1.5,
                label=label_sano if i == 0 else "")
    
    # Plot Team Arrhythmia (Red)
    label_arr = "Team Arrhythmia (Energy)"
    for i in range(mid_point, n_neurons):
        ax.plot(times, v_traces[i], color='firebrick', alpha=0.5, linewidth=1.5,
                label=label_arr if i == mid_point else "")

    # 4. CLINICAL BOUNDARIES
    # Draw the firing thresholds
    ax.axhline(threshold_sano, color='green', linestyle='--', linewidth=1, alpha=0.8, 
               label=f'Thresh Healthy ({threshold_sano}mV)')
    ax.axhline(threshold_arr, color='red', linestyle='--', linewidth=1, alpha=0.8, 
               label=f'Thresh Arrhythmia ({threshold_arr}mV)')

    # 5. STYLING (The "Paper" Look)
    ax.set_title(f"Neuromorphic Diagnosis: {title}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
    
    # Zoom in on the relevant voltage range
    ax.set_ylim(-90, -10)
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_weight_matrices(syn_sano, syn_arr):
    """
    Visualizes the learned synaptic weights as heatmaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    def sparse_to_dense(synapses, n_in=90, n_out=5):
        matrix = np.zeros((n_in, n_out))
        if len(synapses) == 0: return matrix
        
        j_normalized = synapses.j - np.min(synapses.j)
        weights = synapses.w / b2.mV if hasattr(synapses.w, 'dim') else synapses.w
        matrix[synapses.i, j_normalized] = weights
        return matrix

    # Sano Matrix
    w_matrix_sano = sparse_to_dense(syn_sano)
    im1 = ax1.imshow(w_matrix_sano.T, aspect='auto', cmap='Greens', interpolation='nearest')
    ax1.set_title('Learned Weights: Healthy Team', fontsize=11)
    ax1.set_xlabel('Input Neuron ID')
    plt.colorbar(im1, ax=ax1, label='Weight (mV)')

    # Arrhythmia Matrix
    w_matrix_arr = sparse_to_dense(syn_arr)
    im2 = ax2.imshow(w_matrix_arr.T, aspect='auto', cmap='Reds', interpolation='nearest')
    ax2.set_title('Learned Weights: Arrhythmia Team', fontsize=11)
    ax2.set_xlabel('Input Neuron ID')
    plt.colorbar(im2, ax=ax2, label='Weight (mV)')
    
    plt.tight_layout()
    plt.show()

def plot_ecg_validation(raw_signal, fs, spike_times_b2, title="Spike Encoding Validation"):
    """
    Visualizes the original analog ECG signal overlayed with the generated neural spikes.
    Useful for demonstrating the temporal precision of the encoding algorithm.
    """
    # Create time axis for the analog signal
    times_sec = np.arange(len(raw_signal)) / fs
    
    # Handle Brian2 units (strip units if present)
    if hasattr(spike_times_b2, 'dim'):
        spikes_sec = spike_times_b2 / b2.second
    else:
        spikes_sec = spike_times_b2

    plt.figure(figsize=(10, 4))
    
    # Plot Original Analog Signal
    
    plt.plot(times_sec, raw_signal, 'k-', alpha=0.6, label='Real ECG (Analog)')
    
    # Plot Digital Spikes as vertical lines
    # vlines creates a discrete "barcode" effect matching the SNN input
    plt.vlines(spikes_sec, ymin=np.min(raw_signal), ymax=np.max(raw_signal), 
               colors='r', alpha=0.3, linewidth=1, label='SNN Spikes (Digital)')
    
    # Styling
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()