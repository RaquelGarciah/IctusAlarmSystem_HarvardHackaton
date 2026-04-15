import brian2 as b2

def build_network(n_input=90, n_output=10, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8, learning_enabled=True,
                  topology_data=None):
    """
    Constructs a Spiking Neural Network (SNN) for ECG classification.
    Employs a competitive architecture with two neuronal populations 
    specialized in different temporal scales.
    """
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms 
    
    # ELECTROPHYSIOLOGICAL PARAMETERS
    # v_rest: Resting membrane potential (baseline state)
    # v_reset: Post-spike reset potential (hyperpolarization)
    # tau_refrac: Absolute refractory period (prevents unnatural high-frequency bursts)
    # w_max: Synaptic weight ceiling to prevent network saturation
    # w_inhib: Inhibitory weight for the Winner-Take-All competitive mechanism
    v_params = {
        'v_rest': -70 * b2.mV, 
        'v_reset': -80 * b2.mV,
        'tau_refrac': 5 * b2.ms,
        'w_max': 4.0 * b2.mV,
        'w_inhib': 250 * b2.mV
    }

    # MATHEMATICAL MODELING
    # Leaky Integrate-and-Fire (LIF) membrane dynamics.
    # dv/dt: Voltage evolution towards resting potential.
    # tau_m: Membrane time constant defining the "integration window" or memory.
    eqs_lif = '''
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    tau_m : second
    v_thresh : volt 
    '''

    # Spike-Timing-Dependent Plasticity (STDP) model.
    # dapre/dt & dapost/dt: Decay traces to track temporal causality 
    # between pre-synaptic and post-synaptic events.
    eqs_stdp = '''
    w : volt
    dapre/dt = -apre/tau_pre : 1 (clock-driven)
    dapost/dt = -apost/tau_post : 1 (clock-driven)
    '''

    # NEURAL GROUP CONFIGURATION
    # Input Group: Encodes pre-processed ECG data into discrete spike events.
    if spike_indices is None:
        spike_indices, spike_times = [0], [0]*b2.ms
    
    input_group = b2.SpikeGeneratorGroup(n_input, indices=spike_indices, 
                                         times=spike_times, name='Input')
    
    # Output Group: Neurons responsible for information integration and competition.
    output_group = b2.NeuronGroup(n_output, eqs_lif, threshold='v > v_thresh',
                                  reset='v = v_reset', refractory=v_params['tau_refrac'],
                                  method='exact', namespace=v_params, name='Output')
    output_group.v = v_params['v_rest']

    # POPULATION SPECIALIZATION (DUAL-TAU STRATEGY)
    # Subdividing the output group into functional sub-matrices.
    mid = n_output // 2
    team_sano = output_group[:mid]      # "Sharp-shooter" population (Normal rhythm)
    team_arritmia = output_group[mid:]  # "Integrator" population (Arrhythmic patterns)
    
    # Sano Team: Short tau_m for high-precision temporal coincidence detection.
    team_sano.tau_m = 10 * b2.ms       
    team_sano.v_thresh = -55 * b2.mV

    # Arritmia Team: Long tau_m to integrate energy from dispersed or noisy signals.
    team_arritmia.tau_m = 80 * b2.ms  
    team_arritmia.v_thresh = -50 * b2.mV

    # SYNAPTIC CONFIGURATION & LEARNING RULES
    # Defining input-to-output connectivity and learning dynamics.
    
    # Sano Rule: Narrow learning windows (tau_pre/post) for high morphological fidelity.
    sano_cfg = {**v_params, 'tau_pre': 10*b2.ms, 'tau_post': 10*b2.ms, 
                'dA_plus': 0.40*b2.mV, 'dA_minus': 0.05*b2.mV}
    
    synapses_sano = b2.Synapses(
        input_group, team_sano, model=eqs_stdp,
        on_pre='v_post += w; apre += 1; w = clip(w - dA_minus * apost, 0*mV, w_max)' if learning_enabled else 'v_post += w',
        on_post='apost += 1; w = clip(w + dA_plus * apre, 0*mV, w_max)' if learning_enabled else '',
        method='exact', namespace=sano_cfg, name='Syn_Sano'
    )

    # Arritmia Rule: Wide windows to capture low-frequency jitter and noise patterns.
    arr_cfg = {**v_params, 'tau_pre': 60*b2.ms, 'tau_post': 60*b2.ms, 
               'dA_plus': 2.5*b2.mV, 'dA_minus': 0.1*b2.mV}

    synapses_arritmia = b2.Synapses(
        input_group, team_arritmia, model=eqs_stdp,
        on_pre='v_post += w; apre += 1; w = clip(w - dA_minus * apost, 0*mV, w_max)' if learning_enabled else 'v_post += w',
        on_post='apost += 1; w = clip(w + dA_plus * apre, 0*mV, w_max)' if learning_enabled else '',
        method='exact', namespace=arr_cfg, name='Syn_Arritmia'
    )

    # CONNECTIVITY & INITIALIZATION
    # Stochastic connection matrix initialization in the absence of prior topology data.
    if topology_data is None:
        synapses_sano.connect(p=connectivity_prob)
        synapses_arritmia.connect(p=connectivity_prob)
        
        # Initializing weights at a medium-high baseline to encourage initial neural firing.
        initial_w = '2.8 * mV + rand() * 1.0 * mV'
        synapses_sano.w = initial_w
        synapses_arritmia.w = initial_w

    # RECIPROCAL LATERAL INHIBITION
    # Implements a negative feedback loop to enforce "Winner-Take-All" competition.
    # When one population fires, it drains the membrane potential of the rival group.
    inhib_S_A = b2.Synapses(team_sano, team_arritmia, on_pre='v_post -= w_inhib',
                            namespace=v_params, name='Inhib_S_A')
    inhib_S_A.connect()

    # Arritmia-to-Sano inhibition path.
    inhib_A_S = b2.Synapses(team_arritmia, team_sano, on_pre='v_post -= w_inhib',
                            namespace=v_params, name='Inhib_A_S')
    inhib_A_S.connect()

    # INSTRUMENTATION & DATA EXTRACTION 
    # sp_in/sp_out: Monitors spike timings for statistical rate analysis.
    # v_mon: Tracks continuous membrane potential for diagnostic visualization.
    sp_in = b2.SpikeMonitor(input_group)
    sp_out = b2.SpikeMonitor(output_group)
    v_mon = b2.StateMonitor(output_group, 'v', record=True)
    
    net_objects = [input_group, output_group, synapses_sano, synapses_arritmia, 
                   inhib_S_A, inhib_A_S, sp_in, sp_out, v_mon]

    # Optional weight monitors for tracking STDP convergence during training.
    w_mon_s = None
    w_mon_a = None
    if len(synapses_sano) > 0:
        w_mon_s = b2.StateMonitor(synapses_sano, 'w', record=[0], dt=100*b2.ms)
        net_objects.append(w_mon_s)
    if len(synapses_arritmia) > 0:
        w_mon_a = b2.StateMonitor(synapses_arritmia, 'w', record=[0], dt=100*b2.ms)
        net_objects.append(w_mon_a)

    return {
        'net': b2.Network(net_objects),
        'input': input_group,
        'output': output_group,
        'synapses_sano': synapses_sano,
        'synapses_arr': synapses_arritmia,
        'mon_out': sp_out,
        'mon_v': v_mon,
        'n_output': n_output,
        'mon_w_sano': w_mon_s, 
        'mon_w_arr': w_mon_a
    }