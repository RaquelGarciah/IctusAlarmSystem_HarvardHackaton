# 🫀 SNN-Arrhythmia-Detector: Neuromorphic ECG Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Brian2](https://img.shields.io/badge/Simulator-Brian2-green)
![Status](https://img.shields.io/badge/Status-Clinical%20Validation-orange)

A bio-inspired **Spiking Neural Network (SNN)** designed to detect cardiac arrhythmias using **Spike-Timing-Dependent Plasticity (STDP)**. This project bridges the gap between biological signal processing and neuromorphic computing, validating its performance against real-world data from the **MIT-BIH Arrhythmia Database**.

---

## Abstract

Standard ECG analysis relies on digital signal processing or heavy deep learning models (CNN/RNN). This project explores a **neuromorphic approach**, encoding analog ECG signals into discrete spike trains. The network employs a competitive **Dual-Population Architecture** with lateral inhibition, where neurons specialize in detecting either synchronous (Healthy) or asynchronous (Arrhythmic) patterns based on the morphological jitter of the QRS complex.

## Key Features

* **Dual-Tau Architecture:** Implements two competing neuronal populations with distinct membrane time constants ($\tau$):
    * **Healthy Team (Low $\tau$):** Acts as a coincidence detector for precise, rhythmic signals.
    * **Arrhythmia Team (High $\tau$):** Acts as a temporal integrator for chaotic or wide QRS complexes.
* **Unsupervised Learning (STDP):** The network learns to distinguish patterns without explicit labeling during the training phase, utilizing biological Hebbian learning rules.
* **Winner-Take-All Competition:** Implements strong lateral inhibition to force decision-making between the diagnostic populations.
* **Clinical Validation:** Integrated directly with the **MIT-BIH Database** (PhysioNet) to test against real patient records (e.g., Patient 115 vs. Patient 203).

## Pipeline Implementation

In this package, we implemented a custom SNN pipeline using **Brian2** for differential equation modeling and **WFDB** for clinical data acquisition. The main components are:

### 1. Analog-to-Spike Encoding (`src/real_ecg_loader.py`)
Implemented an `ecg_to_spikes` conversion algorithm that transforms continuous voltage signals into temporal spike trains.
* **Signal Processing:** Utilized `wfdb` to fetch raw annotations and signals from PhysioNet.
* **Temporal Coding:** Applied a jitter-injection technique to map QRS morphological variance into precise spike timings, critical for the network's plasticity.

### 2. Neuromorphic Architecture (`src/network.py`)
Designed the core topology using Brian2's `NeuronGroup` and `Synapses` classes.
* **LIF Model:** Defined the differential equations for **Leaky Integrate-and-Fire** neurons with adaptive thresholds.
* **Plasticity:** Implemented the **STDP** learning rule, allowing synaptic weights ($w$) to evolve based on the causal relationship ($\Delta t$) between pre- and post-synaptic spikes.

### 3. Inference & Diagnostics (`src/test_recognition.py`)
Created a deterministic inference engine that freezes the network state for clinical validation.
* **Simulation Management:** Developed a lightweight `SimResult` class to extract telemetry (voltage/time) from Brian2 monitors without memory overhead.
* **Diagnostic Logic:** Implemented a comparison function `compare_recognition` that evaluates the firing rates and membrane potentials of the competing populations to issue a diagnosis.

### 4. Persistence Layer (`src/weight_manager.py`)
Handled the serialization of the trained network state.
* **Serialization:** Implemented `save_weights` and `load_weights` using `pickle` to store synaptic matrices and topology, stripping physical units for storage efficiency and reconstructing them upon loading.

---

## 📂 Project Structure

```text
├── src/
│   ├── main_ecg.py          # Entry point: Orchestrates Training & Validation
│   ├── network.py           # SNN Architecture Definition (LIF Neurons + STDP)
│   ├── real_ecg_loader.py   # Signal Processing: Analog ECG -> Spike Encoding
│   ├── weight_manager.py    # Persistence Layer (Save/Load Synaptic Weights)
│   ├── test_recognition.py  # Inference Engine & Diagnostic Logic
│   └── visualization.py     # Plotting tools for Membrane Potentials
├── saved_weights/           # Serialized network state (.pkl)
├── requirements.txt         # Dependencies
└── README.md
```
Installation
Clone the repository:

Bash:
```
git clone [https://github.com/RaquelGarciah/ICTUS_ALARM_SYSTEM.git](https://github.com/RaquelGarciah/ICTUS_ALARM_SYSTEM.git)
cd Redes-Neuronales
```
Install dependencies: This project relies on Brian2 for simulation and wfdb for clinical data access.
```
```
pip install -r requirements.txt
```
Usage
To run the full pipeline (Training -> Weight Saving -> Clinical Validation):

```
python src/main_ecg.py
```
Phase 1 (Training): The network trains on synthetic generated data to learn the foundational features of normal vs. arrhythmic beats.

Phase 2 (Validation): The system downloads real patient data (MIT-BIH), converts it to spikes, and performs a blind diagnosis.

To test different patients, edit the DATASET CONFIGURATION section in src/main_ecg.py:
```
RECORD_HEALTHY = '100'
RECORD_ARRHYTHMIA = '200'
Results & Visualization
The system provides a real-time clinical report and membrane potential dynamics visualization.
```
<img width="1200" height="600" alt="Patient_100_Healthy"
src="https://github.com/user-attachments/assets/2f127dcb-7e8d-4414-bcc2-a361125970d6" />

<img width="1200" height="600" alt="Patient_200_Arrythmia"
src="https://github.com/user-attachments/assets/aab372d3-223c-4d63-86b0-e01cb7ac0c25" />
```
- Temporal Precision: The jitter-based encoder achieves >95% temporal alignment with cardiologist annotations.
- Diagnostic Logic: Priority-based heuristic minimizes false positives by prioritizing the high-specificity "Healthy" population.

   
