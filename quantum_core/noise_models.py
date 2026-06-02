"""
Noise Models — Modelli di rumore quantistico realistico.

Implementa i principali canali di decoerenza basati su:
- Kraus operator-sum representation (formalismo Lindbladian)
- T1 (energy relaxation / amplitude damping)
- T2 (dephasing / phase damping) 
- Depolarizing channel
- Correlated noise e crosstalk

I quattro profili hardware (superconducting, trapped_ion, silicon_spin,
neutral_atom) usano valori di T1/T2 e fidelity dell'ordine di grandezza di
quelli riportati per le piattaforme reali corrispondenti. Sono pensati come
parametri realistici di riferimento, non come specifiche di un dispositivo preciso.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class QubitParameters:
    """Parametri fisici di un qubit simulato."""
    # Decoherence times (in microsecondi)
    t1: float = 100.0       # Energy relaxation time
    t2: float = 80.0        # Dephasing time  
    t_gate_single: float = 0.030   # Single qubit gate time (30ns)
    t_gate_two: float = 0.100      # Two-qubit gate time (100ns)
    
    # Gate fidelities (basate su paper 2024-2025)
    fidelity_single: float = 0.9999  # Single qubit ~99.9%+
    fidelity_two: float = 0.9990     # Two-qubit ~99.7-99.9%
    
    # Readout error
    readout_0_error: float = 0.01     # P(error|true=0)
    readout_1_error: float = 0.01     # P(error|true=1)
    
    # Leakage (perdita dallo spazio computazionale)
    leakage_rate: float = 0.001
    
    metadata: Dict = field(default_factory=dict)


class NoiseModelLibrary:
    """
    Libreria di modelli di rumore quantistico.
    
    Fornisce operatori di Kraus per i principali canali di decoerenza,
    e parametri realistici basati su hardware reale (superconduttori, trappole ioniche).
    """

    # ─── Realistic Hardware Profiles ──────────────────────────────

    @staticmethod
    def superconducting_transmon() -> QubitParameters:
        """Profilo qubit superconduttore tipo IBM/Samsung."""
        return QubitParameters(
            t1=80.0,           # 80μs (IBM Eagle ~100μs)
            t2=60.0,           # 60μs (typical T2 < T1)
            t_gate_single=0.030,  # 30ns
            t_gate_two=0.150,     # 150ns (two-qubit più lento)
            fidelity_single=0.9995,
            fidelity_two=0.997,
            readout_0_error=0.02,
            readout_1_error=0.03,
        )

    @staticmethod
    def trapped_ion() -> QubitParameters:
        """Profilo qubit intrappolato tipo IonQ/Quantinuum."""
        return QubitParameters(
            t1=1000.0,         # 1ms (molto più lungo dei superconduttori)
            t2=500.0,          # 500μs
            t_gate_single=0.001,   # 1μs (più lento ma più fedele)
            t_gate_two=0.010,        # 10μs
            fidelity_single=0.99999,  # IonQ EQC: >99.99%
            fidelity_two=0.9999,      # Record IonQ 2025
            readout_0_error=0.001,
            readout_1_error=0.001,
        )

    @staticmethod
    def silicon_spin() -> QubitParameters:
        """Profilo qubit di spin nel silicio."""
        return QubitParameters(
            t1=500.0,          # 500μs
            t2=200.0,          # 200μs  
            t_gate_single=0.010,   # 10ns (molto veloce)
            t_gate_two=0.050,        # 50ns
            fidelity_single=0.9998,
            fidelity_two=0.9972,     # ~99.7% come paper 2024-2025
            readout_0_error=0.01,
            readout_1_error=0.01,
        )

    @staticmethod
    def neutral_atom() -> QubitParameters:
        """Profilo qubit neutro atomico tipo QuEra/Atom Computing."""
        return QubitParameters(
            t1=200.0,          # 200μs
            t2=150.0,          # 150μs
            t_gate_single=0.005,   # 5μs
            t_gate_two=0.020,        # 20μs (molti gate paralleli)
            fidelity_single=0.999,
            fidelity_two=0.995,      # ~99.5% median
            readout_0_error=0.015,
            readout_1_error=0.015,
        )

    @staticmethod
    def get_platform(name: str) -> QubitParameters:
        """Ottiene un profilo hardware per nome."""
        platforms = {
            'superconducting': NoiseModelLibrary.superconducting_transmon,
            'trapped_ion': NoiseModelLibrary.trapped_ion,
            'silicon_spin': NoiseModelLibrary.silicon_spin,
            'neutral_atom': NoiseModelLibrary.neutral_atom,
        }
        
        if name not in platforms:
            raise ValueError(f"Unknown platform: {name}. Available: {list(platforms.keys())}")
        
        return platforms[name]()

    # ─── Kraus Operators ────────────────────────────────────────

    @staticmethod
    def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
        """
        Operatori di Kraus per Amplitude Damping (T1 relaxation).
        
        K₀ = [[1, 0], [0, √(1-γ)]]
        K₁ = [[0, √γ], [0, 0]]
        
        γ = 1 - exp(-t_gate / T₁) dove t è il tempo di gate.
        """
        gamma = max(0, min(1, gamma))
        sqrt_1_g = np.sqrt(max(0, 1 - gamma))
        sqrt_g = np.sqrt(gamma)
        
        return [
            np.array([[1, 0], [0, sqrt_1_g]], dtype=np.complex128),
            np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128),
        ]

    @staticmethod
    def phase_damping_kraus(gamma: float) -> List[np.ndarray]:
        """
        Operatori di Kraus per Phase Damping (T2 dephasing).
        
        K₀ = [[√(1-γ), 0], [0, 1]]
        K₁ = [[√γ, 0], [0, 0]]
        
        γ = 1 - exp(-t_gate / T₂) per dephasing puro.
        
        Nota: T2 ≤ 2T1 (relaxation contribuisce anche al dephasing).
        """
        gamma = max(0, min(1, gamma))
        sqrt_1_g = np.sqrt(max(0, 1 - gamma))
        sqrt_g = np.sqrt(gamma)
        
        return [
            np.array([[sqrt_1_g, 0], [0, 1]], dtype=np.complex128),
            np.array([[sqrt_g, 0], [0, 0]], dtype=np.complex128),
        ]

    @staticmethod
    def depolarizing_kraus(p: float) -> List[np.ndarray]:
        """
        Operatori di Kraus per Depolarizing Channel.
        
        ρ → (1-p)ρ + p(I/2)
        
        K₀ = √(1-p)·I,  K₁ = √(p/3)·X,  K₂ = √(p/3)·Y,  K₃ = √(p/3)·Z
        """
        if p < 0 or p > 1:
            raise ValueError("p must be in [0, 1]")
        
        sqrt_1mp = np.sqrt(max(0, 1 - p))
        sqrt_p3 = np.sqrt(p / 3)
        
        I2 = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        return [sqrt_1mp * I2, sqrt_p3 * X, sqrt_p3 * Y, sqrt_p3 * Z]

    @staticmethod
    def combined_t1t2_kraus(t_gate: float, t1: float, t2: float) -> List[np.ndarray]:
        """
        Combina T1 e T2 in un unico canale di Kraus via product channel.

        Canale = PhDamping ∘ AmpDamping, garantisce Σ K†K = I per costruzione.

        γ₁ = 1 - exp(-t/T1)  (amplitude damping)
        γ_φ = 1 - exp(-t/Tφ) dove 1/Tφ = 1/T2 - 1/(2T1)  (pure dephasing)
        """
        exp_t1 = np.exp(-t_gate / t1) if t1 > 0 else 0.0
        exp_t2 = np.exp(-t_gate / t2) if t2 > 0 else 0.0

        gamma_1 = max(0.0, min(1.0, 1.0 - exp_t1))

        # Pure dephasing factor: exp(-t/Tphi) = exp(-t/T2) / exp(-t/(2*T1))
        phi_factor = np.sqrt(max(0.0, exp_t1))  # exp(-t/(2*T1))
        if phi_factor > 1e-15:
            gamma_phi = max(0.0, min(1.0, 1.0 - exp_t2 / phi_factor))
        else:
            gamma_phi = 1.0

        # K1 = |0><0| + sqrt((1-γ1)(1-γφ)) |1><1|
        # K2 = sqrt(γ1) |0><1|
        # K3 = sqrt(γφ(1-γ1)) |1><1|
        # Completeness: K1†K1 + K2†K2 + K3†K3 = I  (exact by construction)
        sqrt_1g1 = np.sqrt(max(0.0, 1.0 - gamma_1))
        sqrt_g1 = np.sqrt(gamma_1)
        sqrt_gphi_1g1 = np.sqrt(max(0.0, gamma_phi * (1.0 - gamma_1)))
        sqrt_1gphi = np.sqrt(max(0.0, 1.0 - gamma_phi))

        K1 = np.array([[1.0, 0.0], [0.0, sqrt_1g1 * sqrt_1gphi]], dtype=np.complex128)
        K2 = np.array([[0.0, sqrt_g1], [0.0, 0.0]], dtype=np.complex128)
        K3 = np.array([[0.0, 0.0], [0.0, sqrt_gphi_1g1]], dtype=np.complex128)

        kraus = [K for K in (K1, K2, K3) if np.any(np.abs(K) > 1e-15)]
        return kraus if kraus else [np.eye(2, dtype=np.complex128)]

    # ─── Error Rate from Gate Time ──────────────────────────────

    @staticmethod
    def error_rate_from_gate_time(t_gate: float, t1: float, t2: float) -> Tuple[float, float]:
        """
        Calcola i tassi di errore da tempi di gate e decoerenza.
        
        Returns:
            (amplitude_damping_prob, phase_damping_prob)
        """
        gamma_1 = 1 - np.exp(-t_gate / max(t1, 1e-9))
        gamma_2 = 1 - np.exp(-t_gate / max(t2, 1e-9))
        
        return (gamma_1, gamma_2)

    @staticmethod
    def effective_fidelity(gamma_amp: float, gamma_phase: float, gate_fidelity: float) -> float:
        """
        Calcola la fedeltà effettiva di un gate includendo decoerenza.
        
        F_eff ≈ F_gate × (1 - γ_decoherence)
        """
        total_error = 1 - gate_fidelity + gamma_amp + gamma_phase / 2
        return max(0, min(1, 1 - total_error))

    # ─── Crosstalk & Correlated Noise ──────────────────────────

    @staticmethod
    def correlated_noise_kraus(qubit_pairs: List[Tuple[int, int]], 
                                correlation_strength: float = 0.05) -> List[np.ndarray]:
        """
        Genera operatori di Kraus per rumore correlato tra coppie di qubit.
        
        Basato su: Phys. Rev. A (May 2026) — multiqubit correlated noise.
        
        Args:
            qubit_pairs: Liste di coppie di qubit con rumore correlato
            correlation_strength: Forza della correlazione (0-1)
        """
        # Modello semplificato: errore congiunto su entrambe le coppie
        K_no_error = np.eye(4, dtype=np.complex128) * np.sqrt(1 - correlation_strength)
        
        # Errori correlati X⊗X, Y⊗Y, Z⊗Z
        XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex128)
        YY = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]], dtype=np.complex128)
        ZZ = np.diag([1, -1, -1, 1]).astype(np.complex128)
        
        sqrt_corr = np.sqrt(correlation_strength / 3)
        
        return [K_no_error, sqrt_corr * XX, sqrt_corr * YY, sqrt_corr * ZZ]

    # ─── Readout Error Model ──────────────────────────────────

    @staticmethod
    def readout_error_matrix(p_0_err: float, p_1_err: float) -> np.ndarray:
        """
        Matrice di probabilità di errore di lettura.
        
        P(measure=0 | true=0) = 1 - p_0_err
        P(measure=1 | true=1) = 1 - p_1_err
        
        Returns matrix: [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]
        """
        return np.array([
            [1 - p_0_err, p_1_err],
            [p_0_err, 1 - p_1_err]
        ], dtype=np.float64)

    @staticmethod
    def apply_readout_error(results: List[int], qubit_params: QubitParameters, 
                           rng: Optional[np.random.Generator] = None) -> List[int]:
        """
        Applica errore di lettura ai risultati di misura.
        
        Args:
            results: Lista di risultati di misura (0 o 1)
            qubit_params: Parametri del qubit
            rng: Generatore casuale opzionale
        
        Returns:
            Risultati con errore di lettura applicato
        """
        if rng is None:
            rng = np.random.default_rng()
        
        noisy_results = []
        
        for result in results:
            if result == 0:
                if rng.random() < qubit_params.readout_0_error:
                    noisy_results.append(1)
                else:
                    noisy_results.append(0)
            else:
                if rng.random() < qubit_params.readout_1_error:
                    noisy_results.append(0)
                else:
                    noisy_results.append(1)
        
        return noisy_results

    # ─── Noise Schedule (per circuito completo) ────────────────

    def get_noise_schedule(self, circuit: List[dict], 
                          qubit_params: Dict[int, QubitParameters]) -> List[dict]:
        """
        Genera un piano di rumore per ogni gate del circuito.
        
        Per ogni gate, calcola i parametri di Kraus basati su:
        - Tempo del gate (T1/T2 decoerenza)
        - Fedeltà intrinseca del gate
        - Crosstalk con qubit vicini
        
        Returns lista di {'gate_index': int, 'kraus_ops': [...], 'qubits': [...] }
        """
        schedule = []
        
        for idx, gate in enumerate(circuit):
            qubits = gate.get('qubits', [])
            
            # Determina il tempo del gate
            if len(qubits) == 1:
                t_gate = max(
                    qubit_params[q].t_gate_single 
                    for q in qubits if q in qubit_params
                )
                params = qubit_params.get(qubits[0])
                
                if params:
                    gamma_amp, gamma_phase = self.error_rate_from_gate_time(
                        t_gate, params.t1, params.t2
                    )
                    
                    kraus_ops = self.combined_t1t2_kraus(t_gate, params.t1, params.t2)
                    
                    schedule.append({
                        'gate_index': idx,
                        'kraus_ops': kraus_ops,
                        'qubits': qubits,
                        'gamma_amp': gamma_amp,
                        'gamma_phase': gamma_phase,
                    })
            
            elif len(qubits) == 2:
                t_gate = max(
                    qubit_params[q].t_gate_two 
                    for q in qubits if q in qubit_params
                )
                
                params_a = qubit_params.get(qubits[0])
                params_b = qubit_params.get(qubits[1])
                
                if params_a and params_b:
                    gamma_amp, gamma_phase = self.error_rate_from_gate_time(
                        t_gate, min(params_a.t1, params_b.t1), 
                                 min(params_a.t2, params_b.t2)
                    )
                    
                    kraus_ops = self.combined_t1t2_kraus(t_gate, params_a.t1, params_a.t2)
                    
                    schedule.append({
                        'gate_index': idx,
                        'kraus_ops': kraus_ops,
                        'qubits': qubits,
                        'gamma_amp': gamma_amp,
                        'gamma_phase': gamma_phase,
                    })
        
        return schedule

    # ─── Noise Metrics ────────────────────────────────────────

    @staticmethod
    def calculate_circuit_error(circuit: List,
                                qubit_params: Dict[int, QubitParameters]) -> Dict[str, float]:
        """
        Stima l'errore totale di un circuito.

        Accumula gli errori dei singoli gate per stimare la fedeltà complessiva.
        """
        total_error = 0
        gate_count = 0

        for inst in circuit:
            gate_name = getattr(inst, 'gate_name', None)
            if gate_name in ('BARRIER', 'MEASURE'):
                continue

            qubits = inst.qubits
            gate_count += 1

            if len(qubits) == 1:
                param = qubit_params.get(qubits[0])
                if param:
                    gamma_amp, gamma_phase = NoiseModelLibrary.error_rate_from_gate_time(
                        param.t_gate_single, param.t1, param.t2
                    )
                    gate_error = (1 - param.fidelity_single) + gamma_amp + gamma_phase / 2
                    total_error += max(0, gate_error)

            elif len(qubits) == 2:
                param_a = qubit_params.get(qubits[0])
                if param_a:
                    gamma_amp, gamma_phase = NoiseModelLibrary.error_rate_from_gate_time(
                        param_a.t_gate_two, param_a.t1, param_a.t2
                    )
                    gate_error = (1 - param_a.fidelity_two) + gamma_amp + gamma_phase / 2
                    total_error += max(0, gate_error)

        circuit_fidelity = np.exp(-total_error)  # Approximation per errori piccoli

        return {
            'total_error_probability': min(total_error, 1.0),
            'estimated_circuit_fidelity': float(circuit_fidelity),
            'num_gates': gate_count,
        }
