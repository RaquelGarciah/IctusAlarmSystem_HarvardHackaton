# -*- coding: utf-8 -*-
"""
Created on Fri Jan 2 20:08:42 2026
@author: ggv16

Module: Main Execution Script
Description: Orchestrates the entire lifecycle of the SNN:
             1. Dynamic Training (Learning Phase with Synthetic Data)
             2. Clinical Validation (Inference Phase with Real MIT-BIH Data)
"""

import brian2 as b2
import numpy as np
import gc

# CUSTOM MODULES
from src.network import build_network
from src.simulation import run_simulation
from src.weight_manager import save_weights 

from src.test_recognition import compare_recognition, print_diagnosis_report
from src.visualization import plot_clinical_validation, plot_ecg_validation

from src.ecg_gen import generate_ecg_data
from src.real_ecg_loader import RealECGLoader

if __name__ == '__main__':
    
    # GLOBAL CONFIGURATION
    N_INPUT = 90
    N_OUTPUT = 10   
    
    # Training Hyperparameters
    NUM_EPOCHS = 30              
    DURATION_TRAIN = 200 * b2.ms  
    DURATION_TEST = 4 * b2.second 
    
    WEIGHTS_FILE = "pattern_ecg_trained.pkl"

    # DATASET CONFIGURATION (CHANGE HERE!)
    RECORD_HEALTHY = '100'      # Options: '100', '101', '115', '123'
    RECORD_ARRHYTHMIA = '203'   # Options: '200', '201', '203', '106' (PVC)

    # PHASE 1: DYNAMIC TRAINING (Simulation)
    print("\n[ PHASE 1: DYNAMIC TRAINING]")
    print(f" Training SNN on synthetic data for {NUM_EPOCHS} epochs...")

    # 1. Build Network (STDP Enabled)
    train_env = build_network(
        n_input=N_INPUT,
        n_output=N_OUTPUT,
        spike_indices=[0],      
        spike_times=[0]*b2.ms,  
        learning_enabled=True,
        connectivity_prob=0.7
    )

    for epoch in range(NUM_EPOCHS):
        current_bpm = np.random.normal(75, 5) 
        
        #  ALTERNATING CURRICULUM 
        if epoch % 2 == 0:
            current_mode = 'healthy'
            print(f"    Epoch {epoch+1} | Teaching:  NORMAL SINUS")
        else:
            current_mode = 'arrhythmia'
            print(f"    Epoch {epoch+1} | Teaching:  ARRHYTHMIA")

        # SYMMETRIC INHIBITORY GATING (PROTECTION MECHANISM) 
        if 'inhib_S_A' in train_env and 'inhib_A_S' in train_env:
            gate_S_to_A = train_env['inhib_S_A'] 
            gate_A_to_S = train_env['inhib_A_S'] 
            
            if current_mode == 'arrhythmia':
                gate_S_to_A.active = False
                gate_A_to_S.active = True 
            else: # healthy
                gate_S_to_A.active = True
                gate_A_to_S.active = False

        # DATA GENERATION 
        indices_train, times_train = generate_ecg_data(
            n_input=N_INPUT,
            duration_ms=DURATION_TRAIN/b2.ms,
            bpm=current_bpm,
            mode=current_mode,
            noise_level=0.1,             
            jitter_healthy=0.005,    
            jitter_arrhythmia=0.020  
        )
        
        # TEMPORAL ADJUSTMENT 
        current_sim_time = train_env['net'].t / b2.second
        times_train_sec = np.array(times_train, dtype=float)
        times_adjusted = (times_train_sec + current_sim_time) * b2.second
        
        # Inject Data
        train_env['input'].set_spikes(indices_train, times_adjusted)

        # RUN EPOCH 
        run_simulation(train_env, duration=DURATION_TRAIN)
        
        # Print weight stats
        w_s = np.mean(train_env['synapses_sano'].w / b2.mV)
        w_a = np.mean(train_env['synapses_arr'].w / b2.mV)
        print(f"       Avg Weights > Healthy: {w_s:.2f} mV | Arrhythmia: {w_a:.2f} mV")

    # 2. SAVE STATE
    print("\n Serializing synaptic weights...")
    meta_data = {'description': 'ECG Training Sim-to-Real', 'epochs': NUM_EPOCHS}
    save_weights(train_env, filename=WEIGHTS_FILE, metadata=meta_data)
    print(" Training complete.")    
    
    # Cleanup memory before Phase 2
    del train_env
    gc.collect() 

    # PHASE 2: CLINICAL VALIDATION (Real World)
    print("\n[PHASE 2: CLINICAL TEST (MIT-BIH)]")

    # 1. Define Parameters for Inference
    inference_params = {
        'n_input': N_INPUT,
        'n_output': N_OUTPUT,
        'tau_sano': 5 * b2.ms,        
        'tau_arritmia': 100 * b2.ms,
        'w_inhib': 800 * b2.mV,        
        
        # Clinical Thresholds
        'v_thresh_sano': -20 * b2.mV,  
        'v_thresh_arr':  -30 * b2.mV,  
    }

    # 2. Load Real Patient Data
    loader = RealECGLoader(n_input=N_INPUT)
    
    # PATIENT A: HEALTHY (VARIABLE RECORD)
    print(f"\n Loading Patient Control (MIT-BIH {RECORD_HEALTHY} - Healthy)...")
    idx_healthy, t_healthy = loader.ecg_to_spikes(
        input_data=RECORD_HEALTHY,
        duration_sec=DURATION_TEST/b2.second
    )
    # Validate encoding quality against ground truth
    loader.validate_detection(t_healthy, RECORD_HEALTHY) 
    
    # PATIENT B: ARRHYTHMIA (VARIABLE RECORD) 
    print(f"\n Loading Patient Test (MIT-BIH {RECORD_ARRHYTHMIA} - Arrhythmia)...")
    idx_arr, t_arr = loader.ecg_to_spikes(
        input_data=RECORD_ARRHYTHMIA,
        duration_sec=DURATION_TEST/b2.second
    )
    loader.validate_detection(t_arr, RECORD_ARRHYTHMIA)
    
    # [OPTIONAL VISUALIZATION] Uncomment to see the Analog-to-Spike conversion
    # raw_sig_arr, _, fs_arr = loader.load_mit_bih_data(RECORD_ARRHYTHMIA, duration_sec=DURATION_TEST/b2.second)
    # plot_ecg_validation(raw_sig_arr, fs_arr, t_arr, title=f"Spike Encoding: Patient {RECORD_ARRHYTHMIA}")
    
    # raw_sig_arr, _, fs_arr = loader.load_mit_bih_data(RECORD_HEALTHY, duration_sec=DURATION_TEST/b2.second)
    # plot_ecg_validation(raw_sig_arr, fs_arr, t_arr, title=f"Spike Encoding: Patient {RECORD_HEALTHY}")
    
    # 3. EXECUTE DIAGNOSIS
    print("\n Running SNN Comparative Diagnosis...")
    stats, res_healthy, res_arr = compare_recognition(
        inference_params,
        pattern_trained=(idx_healthy, t_healthy),      
        pattern_novel=(idx_arr, t_arr)
    )
    
    # 4. REPORTING
    th_s_mv = inference_params['v_thresh_sano'] / b2.mV
    th_a_mv = inference_params['v_thresh_arr'] / b2.mV
    
    print_diagnosis_report(res_healthy, res_arr, 
                           threshold_sano=th_s_mv, 
                           threshold_arr=th_a_mv)
    
    # 5. VISUALIZATION
    print("\n Generating Clinical Plots...")
    
    # Plot Healthy Case
    plot_clinical_validation(res_healthy, 
                             title=f"Patient {RECORD_HEALTHY} (Control)", 
                             threshold_sano=th_s_mv, 
                             threshold_arr=th_a_mv)
                             
    # Plot Arrhythmia Case
    plot_clinical_validation(res_arr, 
                             title=f"Patient {RECORD_ARRHYTHMIA} (Test)", 
                             threshold_sano=th_s_mv, 
                             threshold_arr=th_a_mv)
    
    print("\n✅ PROJECT COMPLETED SUCCESSFULLY.")