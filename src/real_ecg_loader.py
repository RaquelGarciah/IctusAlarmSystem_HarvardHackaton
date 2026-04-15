# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:30:36 2026
@author: ggv16

Module: Real ECG Data Loader & Spike Encoder
Description: Handles the extraction of raw ECG signals from the MIT-BIH Arrhythmia Database,
             performs R-peak detection, and encodes the analog morphology into 
             neuromorphic spike trains using a Jitter-based algorithm.
"""

import wfdb
import numpy as np
import brian2 as b2
from scipy.signal import find_peaks

class RealECGLoader:
    """
    A pipeline for converting real-world ECG time-series data into 
    event-based spike trains suitable for SNN processing.
    """
    
    def __init__(self, n_input=90):
        """
        Args:
            n_input (int): Number of input neurons (channels) available for encoding.
        """
        self.n_input = n_input
        
    def load_mit_bih_data(self, record_name='100', duration_sec=10):
        """
        Downloads or loads a specific record from the PhysioNet MIT-BIH database.
        
        Args:
            record_name (str): The ID of the patient record (e.g., '100', '201').
            duration_sec (float): Length of the signal to extract in seconds.
            
        Returns:
            tuple: (signal_array, annotation_object, sampling_frequency)
        """
        print(f" Downloading/Loading Real ECG: Record {record_name}...")
        try:
            # Load the raw signal (channel 0 is usually MLII lead)
            record = wfdb.rdrecord(record_name, sampfrom=0, pn_dir='mitdb')
            fs = record.fs
            samp_end = int(duration_sec * fs)
            signal = record.p_signal[:samp_end, 0] 
            
            # Load clinical annotations (Ground Truth) for validation purposes
            try:
                annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            except:
                annotation = None
                
            return signal, annotation, fs
            
        except Exception as e:
            print(f" Error loading MIT-BIH data: {e}")
            return None, None, None

    def detect_r_peaks(self, signal, fs):
        """
        Identifies QRS complex locations using a nonlinear energy operator approach.
        Squaring the signal emphasizes high-frequency, high-amplitude events (R-peaks).
        
        Args:
            signal (array): Raw ECG voltage trace.
            fs (float): Sampling frequency.
            
        Returns:
            array: Indices of detected R-peaks.
        """
        # 
        
        # 1. DC Offset Removal (Baseline Centering)
        centered_signal = signal - np.mean(signal)
        
        # 2. Nonlinear Energy Transformation (Square Operator)
        # This makes all peaks positive and suppresses small noise artifacts.
        squared_signal = centered_signal ** 2
        
        # 3. Min-Max Normalization (0 to 1 range)
        sig_min = np.min(squared_signal)
        sig_max = np.max(squared_signal)
        if sig_max - sig_min == 0: return []
        normalized = (squared_signal - sig_min) / (sig_max - sig_min)
        
        # 4. Peak Detection
        # Height threshold: 0.20 (20% of max energy)
        # Minimum distance: 250ms (Physiological limit for refractory period)
        peaks, _ = find_peaks(normalized, height=0.20, distance=int(fs*0.25))
        
        return peaks

    def measure_qrs_width(self, signal, peak_idx, fs):
        """
        Estimates the QRS complex duration using the Full Width at Half Maximum (FWHM) method.
        This metric is crucial for distinguishing narrow (Healthy) vs. wide (Ventricular) beats.
        """
        # Use absolute value to handle inverted peaks (common in PVCs)
        abs_sig = np.abs(signal - np.mean(signal))
        
        # Boundary safety check
        if peak_idx <= 0 or peak_idx >= len(signal) - 1:
            return 0.1 # Default safety value (100ms)
            
        # 1. Determine Peak Amplitude
        peak_amp = abs_sig[peak_idx]
        
        # 2. Define Half-Max Threshold (50%)
        half_height = peak_amp * 0.5
        
        # 3. Scan Left (Onset)
        left = peak_idx
        while left > 0 and abs_sig[left] > half_height:
            left -= 1
            
        # 4. Scan Right (Offset)
        right = peak_idx
        while right < len(signal) - 1 and abs_sig[right] > half_height:
            right += 1
            
        # 5. Convert width to seconds
        width_samples = right - left
        width_seconds = width_samples / fs
        
        return width_seconds

    def ecg_to_spikes(self, input_data, fs=360.0, duration_sec=10):
        """
        Core Encoding Algorithm: Converts analog ECG into spike trains.
        Maps QRS morphology (width) to temporal jitter (spike dispersion).
        
        Args:
            input_data: Either a Record Name (str) or Raw Signal (array).
            
        Returns:
            tuple: (spike_indices, spike_times) for Brian2 input.
        """
        # 
        
        try:
            # INPUT HANDLING 
            if isinstance(input_data, str):
                # Load from database
                loaded_data = self.load_mit_bih_data(input_data, duration_sec)
                signal = loaded_data[0]
                # Default MIT-BIH frequency if not provided
                fs = 360.0 
                
            elif isinstance(input_data, (np.ndarray, list)):
                # Use provided raw array
                signal = np.array(input_data)
                
            else:
                raise ValueError("input_data must be a record name (str) or signal array.")

            # PROCESSING PIPELINE
            
            # 1. R-Peak Detection
            peaks = self.detect_r_peaks(signal, fs)
            
            # 2. Morphology-to-Jitter Encoding
            indices = []
            times = []
            
            for r_peak in peaks:
                # Extract a 100ms window around the beat
                window_samples = int(0.100 * fs)
                start = max(0, r_peak - window_samples)
                end = min(len(signal), r_peak + window_samples)
                
                beat_window = signal[start:end]
                if len(beat_window) < 5: continue
                
                # Calculate Pulse Width (FWHM)
                # (Simplified logic integrated directly for performance)
                peak_val = signal[r_peak]
                half_max = peak_val / 2.0
                crossings = np.where(np.diff(np.sign(beat_window - half_max)))[0]
                
                width_ms = 0
                if len(crossings) >= 2:
                    width_samples = crossings[-1] - crossings[0]
                    width_ms = (width_samples / fs) * 1000
                else:
                    width_ms = 20 # Fallback
                
                # BIOPHYSICAL MAPPING 
                # Narrow QRS (<40ms in this focused window) -> Low Jitter -> Synchronous Input (Healthy)
                # Wide QRS (>40ms) -> High Jitter -> Asynchronous Input (Arrhythmia)
                if width_ms < 40: 
                    jitter = 5e-3 # 5ms dispersion (Synchronous)
                else:              
                    jitter = 25e-3 # 25ms dispersion (Asynchronous)
                
                # Generate a burst of 40 spikes per beat distributed by Jitter
                for i in range(self.n_input):
                    t_spike = (r_peak / fs) + np.random.normal(0, jitter)
                    if 0 <= t_spike < duration_sec:
                        indices.append(i)
                        times.append(t_spike)
            
            # 3. Background Noise Injection (5% Density)
            # Adds robustness to the SNN by simulating sensor noise
            n_noise = int(self.n_input * duration_sec * 0.05) 
            indices.extend(np.random.randint(0, self.n_input, n_noise))
            times.extend(np.random.uniform(0, duration_sec, n_noise))
            
            # 4. Temporal Sorting (Critical for Brian2 Stability)
            # Brian2 requires spike times to be strictly increasing.
            all_times = np.array(times)
            sort_idx = np.argsort(all_times)
            
            indices = np.array(indices)[sort_idx]
            times = all_times[sort_idx]
            
            return indices, times
            
        except Exception as e:
            print(f" Error in ecg_to_spikes: {e}")
            return [], []

    def validate_detection(self, my_times_b2, record_name):
        """
        Validates the SNN's detection against the Clinical Ground Truth (MIT-BIH annotations).
        Includes a clustering algorithm to group neural bursts into single cardiac events.
        
        Args:
            my_times_b2: Spike times generated by the SNN (Brian2 units).
            record_name: MIT-BIH record ID for ground truth comparison.
        """
        try:
            # Unit stripping (Brian2 -> Numpy)
            try:
                my_times = np.array(my_times_b2 / b2.second)
            except:
                my_times = np.array(my_times_b2)
            
            # Load Ground Truth
            record = wfdb.rdrecord(record_name, sampfrom=0, pn_dir='mitdb')
            fs = record.fs
            annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=360*60, pn_dir='mitdb')
            real_times = annotation.sample / fs
            
            # Limit ground truth to simulation duration
            max_sim_time = np.max(my_times) if len(my_times) > 0 else 10
            real_times = real_times[real_times <= max_sim_time]
            
            # INTELLIGENT EVENT CLUSTERING
            # Neurons fire in bursts. We group spikes closer than 50ms into a single "Beat".
            my_beats = []
            if len(my_times) > 0:
                my_times = np.sort(my_times)
                
                current_cluster = [my_times[0]]
                for t in my_times[1:]:
                    if t - current_cluster[-1] < 0.050: # 50ms window
                        current_cluster.append(t)
                    else:
                        # End of cluster. Validate beat density.
                        # Filter: A real beat must trigger at least 15 spikes.
                        if len(current_cluster) >= 15: 
                            my_beats.append(np.mean(current_cluster))
                        
                        current_cluster = [t] # Start new cluster
                
                # Process final cluster
                if len(current_cluster) >= 15:
                    my_beats.append(np.mean(current_cluster))
            
            # METRIC CALCULATION 
            print(f"\n     LABEL VALIDATION (Ground Truth vs. SNN):")
            print(f"       - Physician Labels: {len(real_times)}")
            print(f"       - SNN Detections (>15 spikes): {len(my_beats)}")
            
            # Temporal Coincidence Check (Tolerance Window: 100ms)
            hits = 0
            for t_real in real_times:
                dist = np.min(np.abs(np.array(my_beats) - t_real)) if len(my_beats) > 0 else 999
                if dist < 0.1: 
                    hits += 1
            
            accuracy = (hits / len(real_times)) * 100 if len(real_times) > 0 else 0
            print(f"       - Temporal Accuracy: {accuracy:.1f}%")
            
        except Exception as e:
            print(f"        Validation skipped: {e}")