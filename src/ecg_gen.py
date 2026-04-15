# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
import numpy as np
import brian2 as b2

# Añadimos 'jitter_healthy' y 'jitter_arrhythmia' a los argumentos
def generate_ecg_data(n_input=90, duration_ms=200, bpm=75, mode='healthy', 
                     noise_level=0.05, 
                     jitter_healthy=0.005,    # 5ms por defecto
                     jitter_arrhythmia=0.030): # 30ms por defecto
    """
    Genera spikes sintéticos con jitter configurable.
    """
    # 1. Tiempos base
    interval_sec = 60.0 / bpm
    duration_sec = duration_ms / 1000.0
    beat_times = np.arange(0.1, duration_sec, interval_sec)
    
    indices = []
    times = []
    
    # Rango de neuronas
    signal_neurons = np.arange(0, 40) 

    # 2. Generación de Patrones (USANDO LOS ARGUMENTOS NUEVOS)
    if mode == 'healthy':
        # Usamos el argumento que viene de fuera
        sigma_jitter = jitter_healthy 
        
        for t_beat in beat_times:
            for neuron_idx in signal_neurons:
                spike_t = t_beat + np.random.normal(0, sigma_jitter)
                indices.append(neuron_idx)
                times.append(spike_t)

    elif mode == 'arrhythmia':
        # Usamos el argumento que viene de fuera
        sigma_jitter = jitter_arrhythmia
        
        for t_beat in beat_times:
            if np.random.rand() > 0.2: 
                for neuron_idx in signal_neurons:
                    spike_t = t_beat + np.random.normal(0, sigma_jitter)
                    # Desfase extra para arritmia
                    spike_t += np.random.uniform(0, 0.050)
                    indices.append(neuron_idx)
                    times.append(spike_t)

    # ... (El resto del código de ruido y limpieza se queda IGUAL) ...
    # ... (Copiar la parte de # 3. Ruido y limpieza que ya tenías) ...
    
    # --- BLOQUE DE LIMPIEZA IGUAL QUE ANTES ---
    num_noise = int(n_input * duration_sec * noise_level * 100)
    indices.extend(np.random.randint(0, n_input, num_noise))
    times.extend(np.random.uniform(0, duration_sec, num_noise))
    
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times)
    mask_valid = (all_times >= 0) & (all_times < duration_sec)
    all_indices = all_indices[mask_valid]
    all_times = all_times[mask_valid]
    sort_idx = np.argsort(all_times)
    all_indices = all_indices[sort_idx]
    all_times = all_times[sort_idx]
    dt = 0.0001
    time_steps = (all_times / dt).astype(int)
    unique_id = all_indices * 1e9 + time_steps
    _, unique_idx = np.unique(unique_id, return_index=True)
    final_indices = all_indices[np.sort(unique_idx)]
    final_times = all_times[np.sort(unique_idx)] * b2.second
    
    return final_indices, final_times