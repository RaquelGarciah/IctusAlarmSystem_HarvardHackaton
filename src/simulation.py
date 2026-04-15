# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:13:27 2026

@author: ggv16
"""
import brian2 as b2

def run_simulation(network_dict, duration=1000*b2.ms, pattern_size=5, noise_rate=20*b2.Hz):
    """
    Esta versión ya no necesita set_spikes porque los datos 
    se pasaron al crear el network
    """
    n_total_input = network_dict['input'].N 
    
    print(f"Simulando {duration} con {n_total_input} neuronas de entrada.")
    print(f" -> Patrón en neuronas 0 a {pattern_size-1}")
    print(f" -> Ruido en neuronas {pattern_size} a {n_total_input-1}")

    # Ya no necesitamos generar ni inyectar datos aquí
    # porque ya se hicieron en el main
    
    network_dict['net'].run(duration)
    print("Simulación completada.")