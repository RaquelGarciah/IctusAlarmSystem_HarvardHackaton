# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:11:16 2026

@author: ggv16
"""
import brian2 as b2
import numpy as np

def generate_pattern_data(n_input, duration_ms, pattern_size=3, 
                          noise_rate=20*b2.Hz, dt=0.1):
    """
    Genera datos adaptables a cualquier número de neuronas.
    """
    duration_val = duration_ms  # Duración de un patrón
    pattern_interval = 50       # Cada cuanto se repite el patrón
    dt_pattern = 10              # Separación entre neuronas del patrón
    
    if pattern_size > n_input:
        pattern_size = n_input
        print(f"Aviso: pattern_size reducido a {n_input}")
    
    indices = []
    times = []
    
    # --- 1. PATRÓN (sin repeticiones extras) ---
    n_repetitions = int(duration_val / pattern_interval)
    
    for i in range(n_repetitions):
        base_time = i * pattern_interval
        for p_idx in range(pattern_size):
            indices.append(p_idx)
            t = base_time + p_idx * dt_pattern  #  Simplificado
            t_rounded = np.round(t / dt) * dt
            times.append(t_rounded)
    
    # --- 2. RUIDO (igual que antes) ---
    noise_neurons_list = range(pattern_size, n_input)
    noise_rate_val = float(noise_rate / b2.Hz)
    n_noise_spikes_per_neuron = int(noise_rate_val * duration_val / 1000)
    
    for n_idx in noise_neurons_list:
        rand_times = np.random.rand(n_noise_spikes_per_neuron) * duration_val
        rand_times_rounded = np.round(rand_times / dt) * dt
        unique_times = np.unique(rand_times_rounded)
        
        indices.extend([n_idx] * len(unique_times))
        times.extend(unique_times)
    
    # --- 3. ARRAYS ---
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times)
    
    # --- 4. DEDUPLICAR ---
    spike_pairs = list(zip(all_indices, all_times))
    unique_pairs = list(set(spike_pairs))
    
    if len(unique_pairs) < len(spike_pairs):
        print(f"Aviso: Se eliminaron {len(spike_pairs) - len(unique_pairs)} spikes duplicados")
    
    unique_pairs.sort(key=lambda x: x[1])
    final_indices = np.array([p[0] for p in unique_pairs], dtype=int)
    final_times_vals = np.array([p[1] for p in unique_pairs])
    
    # --- 5. UNIDADES ---
    final_times = final_times_vals * b2.ms
    
    print(f"generate_pattern_data genero:")
    print(f"  - {len(final_indices)} spikes unicos")
    print(f"  - Indices min/max: {final_indices.min()}/{final_indices.max()}")
    print(f"  - Times min/max: {final_times.min()}/{final_times.max()}")
    
    return final_indices, final_times

def generate_pattern_B_different(n_input, duration_ms, pattern_size, dt=0.1):
    """
    Genera un patrón B completamente diferente de A.
    
    Diferencias vs patrón A:
    - Intervalo diferente (25ms vs 50ms)
    - Separación diferente (10ms vs 5ms)
    - Neuronas diferentes (30-60 vs 0-30)
    """
    duration_val = duration_ms
    pattern_interval = 50  # ⭐ Diferente (A usa 50)
    dt_pattern = 25        # ⭐ Diferente (A usa 5)
    pattern_start_neuron = 30  # ⭐ Empezar en neurona 30
    
    indices = []
    times = []
    
    # Generar patrón B
    n_repetitions = int(duration_val / pattern_interval)
    
    for i in range(n_repetitions):
        base_time = i * pattern_interval
        for p_idx in range(pattern_size):
            neuron_id = pattern_start_neuron + p_idx
            if neuron_id >= n_input:  # No pasarse del límite
                break
            
            indices.append(neuron_id)
            t = base_time + p_idx * dt_pattern
            t_rounded = np.round(t / dt) * dt
            times.append(t_rounded)
    
    # Convertir y deduplicar (igual que generate_pattern_data)
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times)
    
    spike_pairs = list(zip(all_indices, all_times))
    unique_pairs = list(set(spike_pairs))
    unique_pairs.sort(key=lambda x: x[1])
    
    final_indices = np.array([p[0] for p in unique_pairs], dtype=int)
    final_times_vals = np.array([p[1] for p in unique_pairs])
    final_times = final_times_vals * b2.ms
    
    print(f"Patrón B generado:")
    print(f"  - {len(final_indices)} spikes")
    print(f"  - Neuronas: {pattern_start_neuron} a {pattern_start_neuron + pattern_size - 1}")
    print(f"  - Intervalo: {pattern_interval}ms (vs 50ms en A)")
    print(f"  - Separación: {dt_pattern}ms (vs 5ms en A)")
    
    return final_indices, final_times