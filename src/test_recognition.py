# -*- coding: utf-8 -*-
"""
Created on Fri Jan 2 20:08:42 2026
@author: ggv16

Module: Clinical Validation & Inference
Description: Executes the diagnostic test on ECG data using the pre-trained SNN.
             Includes signal decoding, network reconstruction, and clinical event analysis.
"""

import brian2 as b2
import numpy as np
from src.weight_manager import load_weights

class SimResult:
    """
    Data container for simulation results.
    Converts Brian2's efficient state monitors into standard Numpy arrays
    for easier post-processing and plotting.
    """
    def __init__(self, t, v):
        self.t = np.array(t)
        self.v = np.array(v)

def ensure_seconds(times):
    """Ensures time arrays have correct Brian2 physical units (seconds)."""
    if not hasattr(times, 'dim'): 
        return times * b2.second
    return times

def compare_recognition(params, pattern_trained, pattern_novel, 
                        topology_save_name='pattern_ecg_trained.pkl',
                        input_gain=1.0):
    """
    Runs a comparative diagnostic test: Healthy Pattern vs. Arrhythmic Pattern.
    
    Args:
        params (dict): Configuration dictionary (time constants, thresholds).
        pattern_trained (tuple): (indices, times) for the Healthy/Control signal.
        pattern_novel (tuple): (indices, times) for the Arrhythmia/Test signal.
        topology_save_name (str): Path to the trained weights file.
        input_gain (float): Scaling factor for input signal strength.
        
    Returns:
        tuple: (stats_dict, result_healthy_obj, result_arrhythmia_obj)
    """
    
    # 1. STRUCTURAL CONFIGURATION
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms
    
    # Parameter Extraction
    N_input = params.get('n_input', 90)
    N_output = params.get('n_output', 10)
    w_inhib = params.get('w_inhib', 250*b2.mV)
    
    # Thresholds (Critical for the competitive logic)
    v_thresh_sano = params.get('v_thresh_sano', -55*b2.mV)     
    v_thresh_arr = params.get('v_thresh_arr', -50*b2.mV)       

    # NEURAL GROUPS 
    input_group = b2.SpikeGeneratorGroup(N_input, [], []*b2.ms, name='Input')
    
    # Leaky Integrate-and-Fire Model
    eqs_lif = '''
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    tau_m : second
    v_rest : volt
    v_thresh : volt
    v_reset : volt
    '''
    
    output_group = b2.NeuronGroup(N_output, eqs_lif,
                                  threshold='v > v_thresh',
                                  reset='v = v_reset',
                                  refractory=2*b2.ms,
                                  method='exact', name='Output')
    
    # Biophysical Properties
    output_group.v_rest = -70*b2.mV
    output_group.v_reset = -80*b2.mV 
        
    # Population Specialization (Dual-Tau)
    n_sano = N_output // 2
    
    # Team Sano: Fast dynamics for coincidence detection
    output_group.tau_m[:n_sano] = params.get('tau_sano', 10*b2.ms)
    output_group.v_thresh[:n_sano] = v_thresh_sano
    
    # Team Arritmia: Slow dynamics for energy integration
    output_group.tau_m[n_sano:] = params.get('tau_arritmia', 80*b2.ms)
    output_group.v_thresh[n_sano:] = v_thresh_arr 
    
    # 2. SYNAPTIC RECONSTRUCTION
    # We explicitly split synapses to map the trained weights correctly.
    
    # Synapse Group 1: Input -> Healthy Team
    syn_sano = b2.Synapses(input_group, output_group[:n_sano], 
                           'w : volt', on_pre='v += w', name='synapses_sano')
                           
    # Synapse Group 2: Input -> Arrhythmia Team
    syn_arr = b2.Synapses(input_group, output_group[n_sano:], 
                          'w : volt', on_pre='v += w', name='synapses_arr')

    # Load Pre-Trained Weights
    # The dictionary keys must match the names used in 'weight_manager.py'
    network_dict = {
        'synapses_sano': syn_sano,
        'synapses_arr': syn_arr
    }
    
    # Reconstruct connectivity and assign weights
    load_weights(network_dict, topology_save_name)
    
    # Apply Signal Gain (Global sensitivity adjustment)
    if input_gain != 1.0:
        syn_sano.w *= input_gain
        syn_arr.w *= input_gain
    
    # 3. LATERAL INHIBITION (COMPETITION)
    # "Winner-Take-All" mechanism: The active team suppresses the other.
    params_inhib = {'w_inhib': w_inhib}
    
    # Healthy suppresses Arrhythmia
    inhib_S_to_A = b2.Synapses(output_group[:n_sano], output_group[n_sano:],
                               'w_in : volt', on_pre='v -= w_in', namespace=params_inhib)
    inhib_S_to_A.connect()
    
    # Arrhythmia suppresses Healthy
    inhib_A_to_S = b2.Synapses(output_group[n_sano:], output_group[:n_sano],
                               'w_in : volt', on_pre='v -= w_in', namespace=params_inhib)
    inhib_A_to_S.connect()
    
    # 4. INSTRUMENTATION
    spikemon = b2.SpikeMonitor(output_group)
    statemon = b2.StateMonitor(output_group, 'v', record=True)
    
    # Network Assembly
    net = b2.Network(input_group, output_group, syn_sano, syn_arr, 
                     inhib_S_to_A, inhib_A_to_S, spikemon, statemon)
    
    # Snapshot the initial state to allow multiple runs
    net.store() 
    
    # TEST CASE 1: HEALTHY PATTERN (CONTROL)
    times_trained = ensure_seconds(pattern_trained[1])
    input_group.set_spikes(pattern_trained[0], times_trained)
    
    if len(times_trained) > 0:
        duration_trained = times_trained[-1] + 0.2*b2.second 
    else:
        duration_trained = 200*b2.ms
        
    net.run(duration_trained)
    
    # Capture results
    res_healthy = SimResult(statemon.t, statemon.v)
    
    # TEST CASE 2: ARRHYTHMIC PATTERN (NOVELTY)
    net.restore() # Reset network state
    
    times_novel = ensure_seconds(pattern_novel[1])
    input_group.set_spikes(pattern_novel[0], times_novel)
    
    if len(times_novel) > 0:
        duration_novel = times_novel[-1] + 0.2*b2.second
    else:
        duration_novel = 1*b2.second
    
    net.run(duration_novel)
    
    res_arrhythmia = SimResult(statemon.t, statemon.v)
    
    # Raw Statistics (for debugging)
    spikes_sano = np.sum(spikemon.count[:n_sano])
    spikes_arritmia = np.sum(spikemon.count[n_sano:])
    
    stats = {
        'total_spikes': spikemon.num_spikes,
        'sano_activity': spikes_sano,
        'arritmia_activity': spikes_arritmia
    }
    
    return stats, res_healthy, res_arrhythmia


def print_diagnosis_report(res_healthy, res_arrhythmia, threshold_sano, threshold_arr):
    """
    Generates a clinical report based on the 'Debounced Event Counting' algorithm.
    Prioritizes the 'Healthy' team's detection due to its high specificity.
    """
    print("\n" + "="*60)
    print("       FINAL CLINICAL REPORT (EVENT COUNTING)        ")
    print("="*60)

    def count_clinical_events(voltage_traces, threshold_mv, refractory_period_ms=150):
        """
        Filters raw neural spikes to count distinct physiological events (Heartbeats).
        Uses a refractory period to debounce high-frequency bursting/noise.
        """
        threshold_val = threshold_mv / 1000.0
        dt = 0.1 # Simulation time step in ms
        refractory_steps = int(refractory_period_ms / dt)
        
        total_events = 0
        
        # Iterate through each neuron in the sub-population
        for i in range(voltage_traces.shape[0]):
            v = voltage_traces[i]
            last_spike_t = -refractory_steps
            
            for t in range(1, len(v)):
                # Detect rising edge crossing
                if v[t-1] < threshold_val and v[t] >= threshold_val:
                    # Enforce refractory period (Debouncing)
                    if (t - last_spike_t) > refractory_steps:
                        total_events += 1
                        last_spike_t = t
                        
        return total_events

    # CASE 1: HEALTHY PATIENT ANALYSIS
    events_green_1 = count_clinical_events(res_healthy.v[:5], threshold_sano)
    events_red_1   = count_clinical_events(res_healthy.v[5:], threshold_arr)

    print(f"\n  CASE 1: HEALTHY PATIENT ")
    print(f"    Green Events (Threshold {threshold_sano} mV): {events_green_1}")
    print(f"    Red Events   (Threshold {threshold_arr} mV): {events_red_1}")

    # DIAGNOSTIC LOGIC:
    # If the Green Team (Precision) detects events, the signal is synchronous (Healthy).
    # The Red Team may also fire due to energy, but Green takes priority.
    
    if events_green_1 >= (events_red_1 * 0.8) and events_green_1 > 0:
        print("   DIAGNOSIS: NORMAL SINUS RHYTHM (HEALTHY)")
    elif events_red_1 > events_green_1:
        print("   DIAGNOSIS: ARRHYTHMIA (False Positive)")
    else:
        print("   INDETERMINATE")

    # CASE 2: ARRHYTHMIA PATIENT ANALYSIS 
    events_green_2 = count_clinical_events(res_arrhythmia.v[:5], threshold_sano)
    events_red_2   = count_clinical_events(res_arrhythmia.v[5:], threshold_arr)

    print(f"\n  CASE 2: ARRHYTHMIA PATIENT")
    print(f"    Green Events (Threshold {threshold_sano} mV): {events_green_2}")
    print(f"    Red Events   (Threshold {threshold_arr} mV): {events_red_2}")
    
    if events_red_2 > events_green_2:
        print("    DIAGNOSIS: ARRHYTHMIA DETECTED")
    elif events_green_2 >= events_red_2:
        print("    DIAGNOSIS: HEALTHY (False Negative)")
    
    print("\n" + "="*60)