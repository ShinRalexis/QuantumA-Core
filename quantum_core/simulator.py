"""
Quantum Simulator Engine — Orchestratore di simulazione completo.

Integra i tre motori di simulazione e i modelli di rumore:
- Statevector simulation (stati puri, ~28 qubit CPU / ~27 GPU)
- Density matrix simulation (stati misti + rumore, ~16 qubit)
- Monte Carlo Wave Function (traiettorie stocastiche, efficiente per molti shot)

Seleziona automaticamente il backend CPU (NumPy) o GPU (PyTorch/CUDA) in
base alla VRAM disponibile, con fallback su CPU. Applica il gate fusion /
cancellation di base tramite QuantumCircuit.optimize() prima dell'esecuzione.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from quantum_core.statevector import StatevectorEngine
from quantum_core.gpu_statevector import GPUStatevectorEngine
from quantum_core.compat import env_bool

try:
    import torch
except Exception:
    torch = None
from quantum_core.density_matrix import DensityMatrixEngine
from quantum_core.gates import QuantumGateLibrary
from quantum_core.noise_models import NoiseModelLibrary, QubitParameters
from quantum_core.circuit import QuantumCircuit


class SimulationMode(Enum):
    """Modalità di simulazione."""
    STATEVECTOR = "statevector"       # Stato puro (senza rumore)
    DENSITY_MATRIX = "density_matrix"  # Stato misto (con rumore)
    MONTE_CARLO = "monte_carlo"       # Shot-based noise (efficiente)


@dataclass
class SimulationResult:
    """Risultati di una simulazione."""
    
    # Risultati di misura
    shot_results: List[int] = field(default_factory=list)
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Metriche fisiche
    final_statevector: Optional[np.ndarray] = None
    final_density_matrix: Optional[np.ndarray] = None
    
    # Statistics
    circuit_fidelity: float = 1.0
    gate_count: int = 0
    simulation_time_ms: float = 0.0
    shots_completed: int = 0
    
    # Errori stimati
    estimated_error_rate: float = 0.0
    noise_summary: Dict[str, float] = field(default_factory=dict)

    def most_likely_states(self, n: int = 5) -> List[Tuple[str, float]]:
        """Ritorna i n stati più probabili."""
        sorted_probs = sorted(
            self.probability_distribution.items(), 
            key=lambda x: x[1], reverse=True
        )[:n]
        
        return [
            (state, round(prob, 6))
            for state, prob in sorted_probs
        ]

    def summary(self) -> str:
        """Genera un riassunto della simulazione."""
        lines = [
            f"=== Simulation Result ===",
            f"Shots completed: {self.shots_completed}",
            f"Circuit fidelity: {self.circuit_fidelity:.6f}",
            f"Simulation time: {self.simulation_time_ms:.3f} ms",
            f"Estimated error rate: {self.estimated_error_rate:.4%}",
            f"",
            f"Top states:",
        ]
        
        for state, prob in self.most_likely_states(5):
            bar = '█' * int(prob * 50) + '░' * (50 - int(prob * 50))
            lines.append(f"  {state}: {prob:.4f} |{bar}|")
        
        return '\n'.join(lines)


class QuantumSimulator:
    """
    Motore di simulazione quantistica completo.
    
    Supporta tre modalità:
    1. STATEVECTOR: Stato puro, veloce, senza rumore (~28 qubit CPU / ~27 GPU)
    2. DENSITY_MATRIX: Stato misto con Kraus operators (~16 qubit)
    3. MONTE_CARLO: traiettorie stocastiche, efficiente per molti shot

    Il circuito viene ottimizzato (gate cancellation / merge di RZ) tramite
    QuantumCircuit.optimize() prima dell'esecuzione.
    """

    def __init__(self, n_qubits: int,
                 mode: SimulationMode = SimulationMode.STATEVECTOR,
                 hardware_profile: str = 'superconducting',
                 backend: str = 'auto',
                 seed: Optional[int] = None):
        """
        Inizializza il simulatore.

        Args:
            n_qubits: Numero di qubit virtuali
            mode: Modalità di simulazione
            hardware_profile: Profilo hardware ('superconducting', 'trapped_ion',
                            'silicon_spin', 'neutral_atom')
            backend: 'auto' | 'cpu' | 'cuda'
            seed: Seme RNG opzionale. Se fornito, rende riproducibili sia la
                  variazione inhomogenea dei parametri di rumore sia il
                  campionamento degli shot (NumPy e, se presente, PyTorch).
        """
        self.n_qubits = n_qubits
        self.mode = mode
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            # Allinea anche il generatore globale legacy usato da np.random.*
            # e da torch per il campionamento, così l'intera pipeline è riproducibile.
            np.random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        self.backend = backend.lower()
        self.gpu_available = bool(
            torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()
        )
        self._statevector_gpu_threshold_qubits = 1
        self._statevector_gpu_preferred_qubits = 4
        if self.backend == 'auto':
            self.backend = self._select_backend(mode)
        elif self.backend == 'cuda' and not self.gpu_available:
            self.backend = 'cpu'
        elif self.backend not in ('cpu', 'cuda'):
            self.backend = 'cpu'
        
        # Gate library e noise models
        self.gate_lib = QuantumGateLibrary()
        self.noise_lib = NoiseModelLibrary()
        
        # Parametri hardware
        self.hardware_profile_name = hardware_profile
        self.qubit_params: Dict[int, QubitParameters] = {}

        # I parametri di rumore per-qubit servono solo nelle modalità con rumore.
        # In STATEVECTOR (stato puro ideale) non vengono usati: evitiamo di
        # generarli, così non si consuma entropia RNG inutilmente.
        if mode != SimulationMode.STATEVECTOR:
            for q in range(n_qubits):
                base_params = self.noise_lib.get_platform(hardware_profile)

                # Variazione leggera per qubit diversi (inhomogeneous noise)
                variation = self._rng.uniform(0.85, 1.15) if n_qubits > 1 else 1.0

                self.qubit_params[q] = QubitParameters(
                    t1=base_params.t1 * variation,
                    t2=base_params.t2 * variation,
                    fidelity_single=base_params.fidelity_single * self._rng.uniform(0.95, 1.0),
                    fidelity_two=base_params.fidelity_two * self._rng.uniform(0.95, 1.0),
                )

        # Engine interno
        self.engine: Optional[Any] = None
        
        if mode == SimulationMode.STATEVECTOR:
            if self.backend == 'cuda' and self.gpu_available:
                # GPU-first: prova ad allocare sulla GPU. Se la VRAM (più la
                # shared memory) non basta nemmeno per lo stato iniziale,
                # intercetta l'OOM e ripiega su CPU senza crashare.
                try:
                    self.engine = GPUStatevectorEngine(n_qubits)
                    self._use_cuda_backend = True
                except Exception as e:
                    if not self._is_oom_error(e):
                        raise
                    self._free_gpu_memory()
                    self.backend = 'cpu'
                    self.engine = StatevectorEngine(n_qubits)
                    self._use_cuda_backend = False
            else:
                self.engine = StatevectorEngine(n_qubits)
                self._use_cuda_backend = False
        elif mode in (SimulationMode.DENSITY_MATRIX, SimulationMode.MONTE_CARLO):
            self.engine = DensityMatrixEngine(n_qubits)
            self._use_cuda_backend = False
        
        # Metrics
        self._simulation_stats = {}

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        """True se l'eccezione è un CUDA out-of-memory (anche su PyTorch vecchi)."""
        oom_type = getattr(torch.cuda, 'OutOfMemoryError', None) if torch is not None else None
        if oom_type is not None and isinstance(exc, oom_type):
            return True
        return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()

    @staticmethod
    def _free_gpu_memory():
        """Libera i buffer CUDA residui dopo un OOM, per non lasciare VRAM occupata."""
        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    # Soglie per la selezione GPU-first.
    # GPU con meno di questa VRAM totale è considerata "scarsa" (es. grafica
    # integrata): per circuiti non banali conviene la CPU.
    _MIN_USEFUL_VRAM_BYTES = 2 * 1024 ** 3          # 2 GB
    # Oltre questo rapporto stato/VRAM totale, la shared memory verrebbe
    # saturata al punto da rendere la GPU più lenta della CPU.
    _MAX_STATE_TO_VRAM_RATIO = 4.0

    def _select_backend(self, mode: SimulationMode) -> str:
        """
        Seleziona il backend con filosofia GPU-first.

        QuantumA Core deve essere scalabile e privilegiare sempre la GPU: la
        CPU è molto più lenta. Quando lo statevector supera la VRAM fisica,
        l'overflow viene assorbito dalla shared/system memory del driver
        (sysmem fallback su Windows/NVIDIA); l'eventuale OutOfMemory è gestito
        a runtime con fallback automatico su CPU (vedi _run_statevector).

        Si sceglie la CPU a priori solo quando:
        - la modalità non è STATEVECTOR (density/monte carlo girano su CPU);
        - non esiste alcuna GPU CUDA;
        - la GPU è troppo piccola per essere utile (VRAM totale < ~2 GB,
          tipicamente grafica integrata);
        - lo stato è assurdamente più grande della VRAM (> 4× la VRAM totale),
          dove la shared memory verrebbe saturata.
        """
        if mode != SimulationMode.STATEVECTOR:
            return 'cpu'
        if not self.gpu_available:
            return 'cpu'
        if self.n_qubits < self._statevector_gpu_threshold_qubits:
            return 'cpu'
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            state_bytes = (1 << self.n_qubits) * 16
            # GPU debole / integrata → CPU per circuiti non banali.
            if total_vram < self._MIN_USEFUL_VRAM_BYTES:
                return 'cpu'
            # Stato troppo grande anche per la shared memory → CPU.
            if state_bytes > total_vram * self._MAX_STATE_TO_VRAM_RATIO:
                return 'cpu'
        except Exception:
            # Nel dubbio si prova comunque la GPU: l'OOM è gestito a runtime.
            return 'cuda'
        return 'cuda'

    def run_circuit(self, circuit: QuantumCircuit, 
                    shots: int = 1024) -> SimulationResult:
        """
        Esegue un circuito quantistico.
        
        Args:
            circuit: Circuito da eseguire
            shots: Numero di misurazioni (per modalità shot-based)
        
        Returns:
            SimulationResult con tutti i risultati
        """
        if circuit.n_qubits != self.n_qubits:
            raise ValueError(f"Circuit qubits ({circuit.n_qubits}) != "
                           f"simulator qubits ({self.n_qubits})")
        
        start_time = time.perf_counter()
        
        # Ottimizza il circuito (gate fusion)
        optimized_circuit = circuit.optimize()
        
        result = SimulationResult()
        result.gate_count = optimized_circuit.total_gates
        
        if self.mode == SimulationMode.STATEVECTOR:
            try:
                result = self._run_statevector(optimized_circuit, shots)
            except Exception as e:
                # GPU-first con rete di sicurezza: se la GPU va in OutOfMemory
                # durante l'esecuzione (VRAM + shared memory esaurite), ripiega
                # su CPU ricostruendo l'engine e rieseguendo da capo.
                if self._use_cuda_backend and self._is_oom_error(e):
                    self._free_gpu_memory()
                    self.backend = 'cpu'
                    self._use_cuda_backend = False
                    self.engine = StatevectorEngine(self.n_qubits)
                    result = self._run_statevector(optimized_circuit, shots)
                    result.noise_summary['gpu_oom_fallback'] = 1.0
                else:
                    raise
            result.noise_summary['backend'] = self.backend
            result.noise_summary['gpu_available'] = float(self.gpu_available)

        elif self.mode == SimulationMode.DENSITY_MATRIX:
            result = self._run_density_matrix(optimized_circuit, shots)
        
        elif self.mode == SimulationMode.MONTE_CARLO:
            result = self._run_monte_carlo(optimized_circuit, shots)

        # I metodi _run_* restituiscono un SimulationResult nuovo: reimposta qui
        # il conteggio gate del circuito ottimizzato (altrimenti resta a 0).
        result.gate_count = optimized_circuit.total_gates

        end_time = time.perf_counter()
        result.simulation_time_ms = (end_time - start_time) * 1000
        
        # Calcola metriche di errore
        if self.mode == SimulationMode.STATEVECTOR:
            result.estimated_error_rate = 0.0
            result.circuit_fidelity = 1.0
        else:
            error_analysis = self.noise_lib.calculate_circuit_error(
                optimized_circuit.instructions, self.qubit_params
            )
            result.estimated_error_rate = error_analysis['total_error_probability']
            result.circuit_fidelity = error_analysis['estimated_circuit_fidelity']
        
        self._simulation_stats = {
            'mode': self.mode.value,
            'n_qubits': self.n_qubits,
            'hardware_profile': self.hardware_profile_name,
            'gate_count': optimized_circuit.total_gates,
            'two_qubit_gates': optimized_circuit.two_qubit_gates,
            'circuit_depth': optimized_circuit.depth,
            'backend': self.backend,
            'gpu_available': self.gpu_available,
        }
        
        return result

    def _run_statevector(self, circuit: QuantumCircuit,
                         shots: int) -> SimulationResult:
        """Esegue la simulazione con statevector."""
        self.engine.reset()
        engine = self.engine
        use_gpu = getattr(self, '_use_cuda_backend', False)

        # Prima esecuzione: applica tutti i gate e salva lo stato finale
        for inst in circuit.instructions:
            if inst.gate_name == 'BARRIER':
                continue
            
            if inst.gate_name == 'MEASURE':
                continue
            
            gate_params = inst.params or {}
            
            try:
                if use_gpu:
                    if len(inst.qubits) == 1:
                        matrix = self.gate_lib.get_matrix(inst.gate_name, **gate_params)
                        engine.apply_gate(matrix, inst.qubits[0])
                    elif len(inst.qubits) == 2:
                        matrix = self.gate_lib.get_two_qubit_gate(inst.gate_name, **gate_params)
                        engine.apply_two_qubit_gate(matrix, inst.qubits[0], inst.qubits[1])
                    else:
                        raise ValueError(f"GPU backend supports max 2 qubits per gate, got {len(inst.qubits)}")
                else:
                    if len(inst.qubits) == 1:
                        engine.apply_gate(inst.gate_name, inst.qubits[0], gate_params)
                    
                    elif len(inst.qubits) == 2:
                        engine.apply_two_qubit_gate(
                            inst.gate_name,
                            inst.qubits[0],
                            inst.qubits[1],
                            gate_params,
                        )
                    
                    else:
                        matrix = self.gate_lib.get_two_qubit_gate(
                            inst.gate_name, **gate_params
                        ) if inst.gate_name != 'CCNOT' else self.gate_lib.ccnot_matrix()
                        
                        engine.apply_custom_gate(matrix, inst.qubits)
            
            except Exception as e:
                # Un OOM della GPU deve propagare, così run_circuit può
                # ripiegare su CPU. Gli altri errori sui singoli gate vengono
                # segnalati senza interrompere il circuito.
                if self._is_oom_error(e):
                    raise
                print(f"Warning: Gate {inst.gate_name} on qubits {inst.qubits}: {e}")

        # Calcola distribuzione di probabilità dallo stato finale
        probabilities = {}
        max_states = min(1 << self.n_qubits, 65536)
        
        if hasattr(engine, 'get_all_probabilities'):
            probs_arr = engine.get_all_probabilities()
            for state_idx in range(max_states):
                prob = float(probs_arr[state_idx])
                if prob > 1e-8:
                    bitstr = format(state_idx, f'0{self.n_qubits}b')
                    probabilities[bitstr] = prob
        else:
            for state_idx in range(max_states):
                prob = engine.probability(state_idx)
                if prob > 1e-8:
                    bitstr = format(state_idx, f'0{self.n_qubits}b')
                    probabilities[bitstr] = prob
        
        # Campiona shots dalla distribuzione di probabilità (piu efficiente)
        shot_results = []
        if probabilities:
            states_list = list(probabilities.keys())
            probs_list = [probabilities[s] for s in states_list]
            total = sum(probs_list)
            probs_norm = [p / total for p in probs_list]
            
            indices = np.random.choice(len(states_list), size=min(shots, 100000), p=probs_norm)
            for idx in indices:
                shot_results.append(int(states_list[idx].replace(' ', ''), 2))
        
        result = SimulationResult(
            shot_results=shot_results[:shots],  # Limita a shots richiesti
            probability_distribution=probabilities,
            final_statevector=engine.to_numpy() if hasattr(engine, 'to_numpy') else engine.amplitudes.copy(),
            shots_completed=len(shot_results),
        )
        
        return result

    def _run_density_matrix(self, circuit: QuantumCircuit,
                            shots: int) -> SimulationResult:
        """Esegue la simulazione con density matrix (con rumore)."""
        self.engine.reset()
        engine = self.engine  # DensityMatrixEngine
        
        for inst_idx, inst in enumerate(circuit.instructions):
            if inst.gate_name == 'BARRIER':
                continue
            
            if inst.gate_name == 'MEASURE':
                continue
            
            gate_matrix = None
            try:
                if len(inst.qubits) == 1:
                    gate_matrix = self.gate_lib.get_matrix(inst.gate_name, **(inst.params or {}))

                    # Applica rumore dopo il gate
                    q = inst.qubits[0]
                    params = self.qubit_params[q]
                    gamma_amp, gamma_phase = self.noise_lib.error_rate_from_gate_time(
                        params.t_gate_single, params.t1, params.t2
                    )
                    
                    # Unitary
                    engine.apply_unitary(gate_matrix, inst.qubits)
                    
                    # Noise (Kraus channels)
                    if gamma_amp > 1e-6:
                        kraus_amp = self.noise_lib.amplitude_damping_kraus(gamma_amp)
                        engine.apply_kraus_channel(kraus_amp, inst.qubits)
                    
                    if gamma_phase > 1e-6:
                        kraus_phase = self.noise_lib.phase_damping_kraus(gamma_phase)
                        engine.apply_kraus_channel(kraus_phase, inst.qubits)
                
                elif len(inst.qubits) == 2:
                    gate_matrix = self.gate_lib.get_two_qubit_gate(
                        inst.gate_name, **inst.params
                    )
                    
                    q0, q1 = inst.qubits
                    params_a = self.qubit_params.get(q0)
                    params_b = self.qubit_params.get(q1)
                    
                    if params_a and params_b:
                        gamma_amp_a, gamma_phase_a = self.noise_lib.error_rate_from_gate_time(
                            params_a.t_gate_two, params_a.t1, params_a.t2
                        )
                        gamma_amp_b, gamma_phase_b = self.noise_lib.error_rate_from_gate_time(
                            params_b.t_gate_two, params_b.t1, params_b.t2
                        )
                        
                        # Unitary + noise
                        engine.apply_unitary(gate_matrix, inst.qubits)
                        
                        gamma_amp = max(gamma_amp_a, gamma_amp_b)
                        gamma_phase = max(gamma_phase_a, gamma_phase_b)
                        
                        if gamma_amp > 1e-6:
                            kraus = self.noise_lib.amplitude_damping_kraus(gamma_amp)
                            # Per i gate a 2 qubit, applichiamo il damping su ciascun qubit separatamente
                            engine.apply_kraus_channel(kraus, [q0])
                            engine.apply_kraus_channel(kraus, [q1])

                elif len(inst.qubits) == 3:
                    # Gate a 3 qubit (es. CCNOT/Toffoli)
                    from quantum_core.gates import QuantumGateLibrary as _GL
                    if inst.gate_name in ('CCNOT', 'Toffoli'):
                        gate_matrix = self.gate_lib.ccnot_matrix()
                    else:
                        raise ValueError(f"Unknown 3-qubit gate: {inst.gate_name}")
                    engine.apply_unitary(gate_matrix, inst.qubits)

            except Exception as e:
                print(f"Warning: {inst.gate_name} on {inst.qubits}: {e}")

        # Misura: campiona tutti gli shot dalla distribuzione finale (senza reset)
        shot_results = []
        dim = 1 << self.n_qubits
        probs_arr = engine.get_probabilities() if hasattr(engine, 'get_probabilities') else None
        if probs_arr is not None:
            outcomes = np.random.choice(range(dim), size=shots, p=probs_arr)
            shot_results = outcomes.tolist()

        probabilities = {}
        if probs_arr is not None:
            for state_idx in range(min(1 << self.n_qubits, 65536)):
                prob = float(probs_arr[state_idx])
                if prob > 1e-8:
                    bitstr = format(state_idx, f'0{self.n_qubits}b')
                    probabilities[bitstr] = prob

        result = SimulationResult(
            shot_results=shot_results[:shots],
            probability_distribution=probabilities,
            shots_completed=len(shot_results),
        )
        
        return result

    def _run_monte_carlo(self, circuit: QuantumCircuit, 
                         shots: int) -> SimulationResult:
        """
        Esegue la simulazione Monte Carlo Wave Function.

        Simula traiettorie pure indipendenti applicando il rumore in modo
        stocastico (un campionamento di Kraus per gate) e media sui N shot.
        Più efficiente della density matrix completa quando servono molti shot.
        """
        engine = self.engine
        
        all_shot_results = []
        
        for shot_idx in range(shots):
            # Reset per ogni shot (ogni traiettoria è indipendente)
            if shot_idx > 0:
                engine.reset()
            
            # Esegue il circuito con rumore Monte Carlo
            for inst in circuit.instructions:
                if inst.gate_name == 'BARRIER':
                    continue
                
                if inst.gate_name == 'MEASURE':
                    continue
                
                gate_matrix = None
                if len(inst.qubits) == 1:
                    gate_matrix = self.gate_lib.get_matrix(inst.gate_name, **(inst.params or {}))
                elif len(inst.qubits) == 2:
                    gate_matrix = self.gate_lib.get_two_qubit_gate(inst.gate_name, **(inst.params or {}))
                elif len(inst.qubits) == 3 and inst.gate_name in ('CCNOT', 'Toffoli'):
                    gate_matrix = self.gate_lib.ccnot_matrix()

                if gate_matrix is not None:
                    engine.apply_unitary(gate_matrix, inst.qubits)

                    # Monte Carlo noise sampling: applica il rumore su OGNI qubit
                    # coinvolto nel gate, ciascuno con i propri parametri T1/T2.
                    is_multi = len(inst.qubits) >= 2
                    for q in inst.qubits:
                        params = self.qubit_params[q]
                        t_gate = params.t_gate_two if is_multi else params.t_gate_single
                        kraus_ops = self.noise_lib.combined_t1t2_kraus(
                            t_gate, params.t1, params.t2
                        )
                        engine = engine.monte_carlo_shot(kraus_ops, qubits=[q])

            # Misura al termine dello shot: campiona dalla distribuzione completa
            probs = engine.get_probabilities()
            dim = 1 << self.n_qubits
            outcome = int(np.random.choice(range(dim), p=probs))
            all_shot_results.append(outcome)

        probabilities = {}
        for state_idx in range(min(1 << self.n_qubits, 65536)):
            prob = float(engine.get_probabilities()[state_idx]) if hasattr(engine, 'get_probabilities') else 0
            if prob > 1e-8:
                bitstr = format(state_idx, f'0{self.n_qubits}b')
                probabilities[bitstr] = prob

        result = SimulationResult(
            shot_results=all_shot_results[:shots],
            probability_distribution=probabilities,
            shots_completed=len(all_shot_results),
        )
        
        return result

    # ─── Pre-built Circuits ──────────────────────────────────────

    def run_bell_state(self, qubit_a: int = 0, qubit_b: int = 1, 
                       shots: int = 1024) -> SimulationResult:
        """Esegue un circuito Bell state."""
        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(qubit_a)
        circuit.cx(qubit_a, qubit_b)
        
        return self.run_circuit(circuit, shots)

    def run_ghz_state(self, shots: int = 1024) -> SimulationResult:
        """Esegue un circuito GHZ state."""
        if self.n_qubits < 3:
            raise ValueError("GHZ requires at least 3 qubits")
        
        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(0)
        for i in range(1, self.n_qubits):
            circuit.cx(0, i)
        
        return self.run_circuit(circuit, shots)

    def run_grover(self, n_iterations: int = 1, 
                   target_state: Optional[int] = None,
                   shots: int = 1024) -> SimulationResult:
        """Esegue l'algoritmo di Grover direttamente sullo statevector per massima efficienza e precisione."""
        import time
        start_time = time.perf_counter()
        
        target = target_state if target_state is not None else (1 << self.n_qubits) - 1
        N = 1 << self.n_qubits
        
        if self.mode != SimulationMode.STATEVECTOR:
            raise ValueError("L'algoritmo di Grover è supportato in modo nativo solo in modalità statevector.")
            
        engine = self.engine
        amp = 1.0 / np.sqrt(N)
        
        if hasattr(engine, 'amplitudes'):
            # Usa PyTorch se GPU, altrimenti NumPy
            is_torch = hasattr(engine.amplitudes, 'fill_')
            if is_torch:
                engine.amplitudes.fill_(amp + 0j)
            else:
                engine.amplitudes.fill(amp + 0j)
                
            for _ in range(n_iterations):
                # Oracle: Inversione di fase sul target
                engine.amplitudes[target] *= -1
                
                # Diffusion Operator: Inversione rispetto alla media
                mean = engine.amplitudes.mean()
                engine.amplitudes = 2 * mean - engine.amplitudes
        
        # Calcola distribuzione
        probabilities = {}
        if hasattr(engine, 'get_all_probabilities'):
            probs_arr = engine.get_all_probabilities()
        else:
            probs_arr = np.abs(engine.amplitudes)**2
            probs_arr = probs_arr / np.sum(probs_arr)
            
        for state_idx in range(min(N, 65536)):
            prob = float(probs_arr[state_idx])
            if prob > 1e-8:
                bitstr = format(state_idx, f'0{self.n_qubits}b')
                probabilities[bitstr] = prob
                
        # Campionamento degli shot
        shot_results = []
        if probabilities:
            states_list = list(probabilities.keys())
            probs_list = [probabilities[s] for s in states_list]
            total = sum(probs_list)
            probs_norm = [p / total for p in probs_list]
            
            indices = np.random.choice(len(states_list), size=min(shots, 100000), p=probs_norm)
            shot_results = [int(states_list[idx], 2) for idx in indices]

        end_time = time.perf_counter()
        
        result = SimulationResult(
            shot_results=shot_results[:shots],
            probability_distribution=probabilities,
            final_statevector=engine.to_numpy() if hasattr(engine, 'to_numpy') else engine.amplitudes.copy(),
            shots_completed=len(shot_results),
            simulation_time_ms=(end_time - start_time) * 1000,
            circuit_fidelity=1.0,
            gate_count=n_iterations * (self.n_qubits * 2 + 1) # Stima del numero di gate teorici
        )
        return result

    # ─── Analysis Tools ──────────────────────────────────────

    def bloch_sphere(self, qubit_idx: int) -> Tuple[float, float, float]:
        """Calcola le coordinate di Bloch per un qubit."""
        if hasattr(self.engine, 'get_bloch_vector'):
            return self.engine.get_bloch_vector(qubit_idx)
        
        elif isinstance(self.engine, DensityMatrixEngine):
            return self.engine.bloch_coordinates(qubit_idx)
        
        raise ValueError("Bloch sphere requires active simulation")

    def entanglement_entropy(self, partition: List[int]) -> float:
        """
        Calcola l'entropia di entanglement S(A) = -Tr(ρ_A log₂ ρ_A)
        per la partizione A = partition (bipartita rispetto al complemento).
        """
        if not isinstance(self.engine, (StatevectorEngine, GPUStatevectorEngine)):
            raise ValueError("Entanglement entropy requires statevector mode")

        n = self.n_qubits
        if hasattr(self.engine, 'to_numpy'):
            psi = self.engine.to_numpy()  # GPU → numpy
        else:
            psi = self.engine.amplitudes  # shape (2^n,)
        dim_a = 1 << len(partition)
        dim_b = 1 << (n - len(partition))

        # Costruisci il vettore di stato riordinato come matrice (dim_a × dim_b)
        # usando la mappatura: indice globale → (indice_A, indice_B)
        other = [q for q in range(n) if q not in partition]
        psi_matrix = np.zeros((dim_a, dim_b), dtype=np.complex128)
        for idx in range(1 << n):
            ia = sum(((idx >> (n - 1 - q)) & 1) << (len(partition) - 1 - i)
                     for i, q in enumerate(partition))
            ib = sum(((idx >> (n - 1 - q)) & 1) << (len(other) - 1 - i)
                     for i, q in enumerate(other))
            psi_matrix[ia, ib] = psi[idx]

        # SVD → valori singolari → entropia di von Neumann
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
        probs = singular_values ** 2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def get_stats(self) -> Dict[str, Any]:
        """Ritorna le statistiche della simulazione corrente."""
        stats = dict(self._simulation_stats)
        
        if self.engine:
            stats['state_purity'] = (
                float(self.engine.get_purity()) 
                if hasattr(self.engine, 'get_purity') else None
            )
            stats['gate_count_total'] = getattr(self.engine, 'gate_count', 0)
        
        return stats

    def __repr__(self):
        return (f"QuantumSimulator(n_qubits={self.n_qubits}, "
                f"mode={self.mode.value}, "
                f"profile='{self.hardware_profile_name}')")
