# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:19:39 2026
@author: ggv16

Module: Unit Testing Suite
Description: Validates core functionality of the SNN pipeline including:
             1. Weight persistence (Serialization/Deserialization).
             2. ECG Signal Encoding and temporal sorting.
"""

import numpy as np
import os
import sys
import brian2 as b2

# Path hack to ensure 'src' is discoverable when running from the 'tests' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.weight_manager import save_weights, load_weights, get_weights_dir
from src.real_ecg_loader import RealECGLoader

#  TEST 1: Weight Persistence Roundtrip (Save -> Load -> Verify)

def test_weight_manager_roundtrip():
    """
    Validates that synaptic weights are correctly saved to disk and restored 
    with high numerical precision. Checks the Dual-Population structure.
    """
    # 1. Test Environment Setup (Brian2)
    b2.start_scope()
    
    # Define dummy neuron groups (Required by Brian2 to link synapses)
    # Note: Threshold and reset are mandatory even if not simulating
    G_in = b2.NeuronGroup(10, 'v : volt', threshold='v > 1*volt', reset='v=0*volt', method='exact')
    G_out = b2.NeuronGroup(10, 'v : volt', method='exact')
    
    # Create two synapse groups to mimic the actual architecture (Healthy vs Arrhythmia)
    S_sano = b2.Synapses(G_in, G_out, 'w : volt', on_pre='v += w')
    S_sano.connect(p=0.5)
    
    S_arr = b2.Synapses(G_in, G_out, 'w : volt', on_pre='v += w')
    S_arr.connect(p=0.5)
    
    # Assign specific random values to verify restoration later
    orig_w_sano = np.random.rand(len(S_sano)) * 10 * b2.mV
    orig_w_arr = np.random.rand(len(S_arr)) * 10 * b2.mV
    
    S_sano.w = orig_w_sano
    S_arr.w = orig_w_arr
    
    # Package objects as expected by weight_manager.py
    network_dict = {
        'synapses_sano': S_sano,
        'synapses_arr': S_arr
    }
    test_filename = "test_weights_pytest.pkl"
    
    # 2. Execute Save
    print(f"\n[TEST] Saving dummy weights to: {test_filename}")
    save_weights(network_dict, filename=test_filename)
    
    # 3. Corrupt/Reset current memory
    # We zero out the weights to ensure the subsequent 'load' actually works
    S_sano.w = 0 * b2.mV
    S_arr.w = 0 * b2.mV
    
    # Verify corruption worked (tolerance for float precision)
    assert np.all(np.abs(S_sano.w) < 1e-9 * b2.volt), "Setup Error: Weights were not reset."
    
    # 4. Execute Load
    print("[TEST] Restoring weights from disk...")
    load_weights(network_dict, filename=test_filename)
    
    # 5. Assertions (Numerical Verification)
    # Strip units for safe numpy comparison
    def strip_units(arr):
        return np.array(arr / b2.mV) if hasattr(arr, 'dim') else np.array(arr)

    # Check Healthy Team Weights
    np.testing.assert_allclose(
        strip_units(S_sano.w), 
        strip_units(orig_w_sano), 
        rtol=1e-5, 
        err_msg="Mismatch in 'Healthy' weights restoration."
    )
    
    # Check Arrhythmia Team Weights
    np.testing.assert_allclose(
        strip_units(S_arr.w), 
        strip_units(orig_w_arr), 
        rtol=1e-5, 
        err_msg="Mismatch in 'Arrhythmia' weights restoration."
    )
    
    print(" Weight Test: SUCCESS. Physical magnitudes preserved correctly.")
    
    # Cleanup: Remove temporary test file
    filepath = os.path.join(get_weights_dir(), test_filename)
    if os.path.exists(filepath):
        os.remove(filepath)

#  TEST 2: ECG Encoder & Temporal Sorting

def test_ecg_encoder_sorting():
    """
    Validates the analog-to-spike conversion pipeline.
    CRITICAL: Ensures spike times are strictly monotonic (required by Brian2).
    """
    # 1. Configuration
    n_input = 40
    # Initialize without Brian2 simulation overhead
    loader = RealECGLoader(n_input=n_input)
    
    fs = 100.0 
    duration = 1.0 
    t = np.linspace(0, duration, int(fs*duration))
    # Generate a synthetic sine wave to simulate a signal
    signal = np.sin(2 * np.pi * 5 * t) 
    
    # 2. Execute Encoder
    print("\n[TEST] Running Encoder: Analog ECG -> Spike Train...")
    indices, times = loader.ecg_to_spikes(signal, fs, duration)
    
    # 3. Unit Handling
    try:
        times_sec = np.array(times / b2.second)
    except:
        times_sec = np.array(times)
        
    indices = np.array(indices)
    
    # 4. Assertions
    assert len(times_sec) > 0, "Encoder failed to generate any spikes."
    
    # Check Monotonicity (Sorting)
    # Brian2 crashes if spike times are not sorted [t0 <= t1 <= t2...]
    is_sorted = np.all(np.diff(times_sec) >= 0)
    assert is_sorted, " CRITICAL ERROR: Spike times are not sorted."
    
    # Check Bounds
    assert np.min(times_sec) >= 0, "Negative spike times detected."
    assert np.max(indices) < n_input, "Neuron indices exceed input layer size."
    
    print(f" Encoder Test: SUCCESS. Generated {len(times_sec)} sorted spikes.")

if __name__ == "__main__":
    # Manual execution block
    test_weight_manager_roundtrip()
    test_ecg_encoder_sorting()