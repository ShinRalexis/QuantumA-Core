"""
Density Matrix Engine — Simulazione di stati misti e rumore realistico.

Implementa la rappresentazione della matrice densità ρ per:
- Stati misti (decoerenza, thermal noise)
- Kraus operator-sum representation (formalismo Lindbladian)
- Monte Carlo Wave Function come alternativa più leggera in memoria

La matrice densità occupa 2^(2n) elementi complessi, quindi il costo di
memoria è doppio in esponente rispetto allo statevector: pratico fino a ~16 qubit.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from quantum_core.statevector import StatevectorEngine


class DensityMatrixEngine:
    """
    Motore di simulazione con matrice densità.
    
    La matrice densità ρ ha dimensione 2^n × 2^n.
    Supporta operazioni unitarie e canali di rumore (Kraus operators).
    
    Memory: O(4^n) complessi → ~32GB per n=16, ~512GB per n=18
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits  # 2^n
        
        # Matrice densità ρ (Hermitiana, semi-definita positiva)
        self.density_matrix = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Stato iniziale: |0⟩⟨0|
        self.density_matrix[0, 0] = 1.0
        
        # Performance tracking
        self.gate_count = 0

    def reset(self):
        """Resetta a ρ = |0⟩⟨0|."""
        self.density_matrix.fill(0)
        self.density_matrix[0, 0] = 1.0
        self.gate_count = 0

    @property
    def is_valid_density_matrix(self) -> bool:
        """Verifica proprietà della matrice densità."""
        # Hermitiana?
        if not np.allclose(self.density_matrix, self.density_matrix.conj().T, atol=1e-12):
            return False
        
        # Traccia = 1?
        trace = np.trace(self.density_matrix)
        if abs(trace.real - 1.0) > 1e-6:
            return False
        
        # Semi-definita positiva? (autovalori ≥ 0)
        eigenvalues = np.linalg.eigvalsh(self.density_matrix)
        return bool(np.all(eigenvalues >= -1e-10))

    def normalize_trace(self):
        """Normalizza la traccia a 1."""
        trace = np.trace(self.density_matrix)
        if abs(trace) > 1e-15:
            self.density_matrix /= trace

    # ─── State Preparation ────────────────────────────────────────

    def from_statevector(self, statevector: np.ndarray):
        """Crea la matrice densità da un vettore di stato puro."""
        self.density_matrix = np.outer(statevector, statevector.conj())

    def get_purity(self) -> float:
        """Purity Tr(ρ²). Purezza=1 per stati puri, <1 per misti."""
        purity = np.real(np.trace(self.density_matrix @ self.density_matrix))
        return float(purity)

    # ─── Unitary Operations ──────────────────────────────────────

    def apply_unitary(self, matrix: np.ndarray, qubits: List[int]):
        """
        Applica un'operazione unitaria: ρ → UρU†.
        
        Versione pragmatica: espande l'operatore locale e usa
        la forma standard della trasformazione per ridurre i path legacy.
        """
        if len(qubits) == 1:
            U = self._expand_single_qubit_unitary(matrix, qubits[0])
        elif len(qubits) == 2:
            U = self._expand_two_qubit_unitary(matrix, qubits[0], qubits[1])
        elif len(qubits) == 3:
            U = self._expand_three_qubit_unitary(matrix, qubits[0], qubits[1], qubits[2])
        else:
            raise ValueError("apply_unitary supports max 3 qubits")

        self.density_matrix = U @ self.density_matrix @ U.conj().T
        self.gate_count += 1

    def apply_simple_unitary(self, matrix: np.ndarray, qubits: List[int]):
        """Versione semplificata per unitari (single e two-qubit)."""
        self.apply_unitary(matrix, qubits)

    def _apply_single_qubit_dm(self, U: np.ndarray, target: int):
        """Compat layer legacy: usa apply_unitary."""
        self.apply_unitary(U, [target])

    def _apply_two_qubit_dm(self, U: np.ndarray, q1: int, q2: int):
        """Compat layer legacy: usa apply_unitary."""
        self.apply_unitary(U, [q1, q2])

    # ─── Noise Channels (Kraus Operators) ────────────────────────

    def apply_kraus_channel(self, kraus_operators: List[np.ndarray], qubits: Optional[List[int]] = None):
        """
        Applica un canale di rumore tramite operatori di Kraus.
        
        ρ → Σ_k K_k ρ K_k†
        
        Args:
            kraus_operators: Lista di matrice di Kraus {K_k} che soddisfano Σ K_k†K_k = I
            qubits: Lista di qubit target. Se specificato, espande gli operatori.
        """
        new_rho = np.zeros_like(self.density_matrix)
        
        for K in kraus_operators:
            if qubits is not None:
                if len(qubits) == 1:
                    K = self._expand_single_qubit_unitary(K, qubits[0])
                elif len(qubits) == 2:
                    K = self._expand_two_qubit_unitary(K, qubits[0], qubits[1])
                elif len(qubits) == 3:
                    K = self._expand_three_qubit_unitary(K, qubits[0], qubits[1], qubits[2])

            # ρ → KρK†
            new_rho += K @ self.density_matrix @ K.conj().T
        
        self.density_matrix = new_rho
        self.normalize_trace()

    def apply_amplitude_damping(self, gamma: float):
        """
        Amplitude Damping (T1 relaxation).
        
        Kraus operators:
            K0 = [[1, 0], [0, sqrt(1-gamma)]]
            K1 = [[0, sqrt(gamma)], [0, 0]]
        
        gamma = 1 - exp(-t/T1) dove t è il tempo di gate.
        """
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma must be in [0, 1]")
        
        sqrt_1_g = np.sqrt(max(0, 1 - gamma))
        sqrt_g = np.sqrt(max(0, gamma))
        
        K0 = np.array([[1, 0], [0, sqrt_1_g]], dtype=np.complex128)
        K1 = np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128)
        
        self.apply_kraus_channel([K0, K1])

    def apply_phase_damping(self, gamma: float):
        """
        Phase Damping (T2 dephasing).
        
        Kraus operators:
            K0 = [[1, 0], [0, sqrt(1-gamma)]]
            K1 = [[sqrt(gamma), 0], [0, 0]]
        """
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma must be in [0, 1]")
        
        sqrt_1_g = np.sqrt(max(0, 1 - gamma))
        sqrt_g = np.sqrt(max(0, gamma))
        
        K0 = np.array([[sqrt_1_g, 0], [0, 1]], dtype=np.complex128)
        K1 = np.array([[sqrt_g, 0], [0, 0]], dtype=np.complex128)

        self.apply_kraus_channel([K0, K1])

    def apply_depolarizing(self, p: float):
        """
        Depolarizing channel.
        
        ρ → (1-p)ρ + p(I/2) per single qubit
           → (1-p)ρ + p(I/4) per two qubits
        
        Kraus operators: √(p/3)·X, √(p/3)·Y, √(p/3)·Z, √(1-p)·I
        """
        if p < 0 or p > 1:
            raise ValueError("p must be in [0, 1]")
        
        sqrt_p3 = np.sqrt(p / 3) if p > 0 else 0
        sqrt_1mp = np.sqrt(max(0, 1 - p))
        
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        
        K_list = [sqrt_1mp * I2, sqrt_p3 * X, sqrt_p3 * Y, sqrt_p3 * Z]
        self.apply_kraus_channel(K_list)

    # ─── Measurement on Density Matrix ────────────────────────────

    def measure(self, target_qubit: int, shots: int = 1) -> List[int]:
        """
        Misura sul formalismo della matrice densità.
        
        Usa le probabilità di Born calcolate dalla diagonale ridotta.
        """
        if target_qubit < 0 or target_qubit >= self.n_qubits:
            raise ValueError(f"Qubit index out of range: {target_qubit}")

        results = []
        bit_pos = self.n_qubits - 1 - target_qubit
        
        for _ in range(shots):
            prob_1 = 0.0
            for i in range(self.dim):
                if (i >> bit_pos) & 1:
                    prob_1 += float(np.real(self.density_matrix[i, i]))
            prob_1 = max(0.0, min(1.0, prob_1))
            results.append(int(np.random.random() < prob_1))

        return results

    def expectation_value(self, observable: np.ndarray, qubits: List[int]) -> float:
        """Calcola ⟨A⟩ = Tr(ρ·A) per un osservabile."""
        # Espandi l'osservabile sullo spazio completo
        n = self.n_qubits
        
        if len(qubits) == 1:
            # Costruisci A ⊗ I⊗...⊗I
            expanded = self._expand_operator(observable, qubits[0])
            return float(np.real(np.trace(expanded @ self.density_matrix)))
        
        elif len(qubits) == 2:
            op_1 = observable
            for q in qubits[1:]:
                op_1 = np.kron(op_1, np.eye(2))
            
            # Riordina per i qubit corretti
            return float(np.real(np.trace(op_1 @ self.density_matrix)))

    def _expand_single_qubit_unitary(self, op: np.ndarray, target_qubit: int) -> np.ndarray:
        """Espande un gate single-qubit allo spazio completo."""
        n = self.n_qubits
        result = np.array([[1]], dtype=np.complex128)
        for q in range(n):
            result = np.kron(result, op if q == target_qubit else np.eye(2, dtype=np.complex128))
        return result

    def _expand_two_qubit_unitary(self, op: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """
        Espande un gate 4x4 sui qubit (q1, q2) allo spazio 2^n × 2^n.

        Funziona per qubit arbitrari, anche non adiacenti. La matrice op
        è in base computazionale |q1 q2⟩ con ordering (q1=MSB, q2=LSB).
        """
        if q1 == q2:
            raise ValueError("Distinct qubits required")
        n = self.n_qubits
        dim = 1 << n

        bit_a = n - 1 - q1
        bit_b = n - 1 - q2
        mask_a = 1 << bit_a
        mask_b = 1 << bit_b

        expanded = np.zeros((dim, dim), dtype=np.complex128)

        for out_idx in range(dim):
            oa = (out_idx >> bit_a) & 1
            ob = (out_idx >> bit_b) & 1
            row = (oa << 1) | ob
            base = out_idx & ~mask_a & ~mask_b

            for col in range(4):
                ia = (col >> 1) & 1
                ib = col & 1
                in_idx = base | (ia << bit_a) | (ib << bit_b)
                expanded[out_idx, in_idx] = op[row, col]

        return expanded

    def _expand_three_qubit_unitary(self, op: np.ndarray, q0: int, q1: int, q2: int) -> np.ndarray:
        """
        Espande un gate 8x8 sui qubit (q0, q1, q2) allo spazio 2^n × 2^n.
        La matrice op è in base |q0 q1 q2⟩ con q0=MSB.
        """
        if q0 == q1 or q1 == q2 or q0 == q2:
            raise ValueError("Distinct qubits required")
        n = self.n_qubits
        dim = 1 << n
        bit0 = n - 1 - q0
        bit1 = n - 1 - q1
        bit2 = n - 1 - q2
        mask0 = 1 << bit0
        mask1 = 1 << bit1
        mask2 = 1 << bit2

        expanded = np.zeros((dim, dim), dtype=np.complex128)

        for out_idx in range(dim):
            o0 = (out_idx >> bit0) & 1
            o1 = (out_idx >> bit1) & 1
            o2 = (out_idx >> bit2) & 1
            row = (o0 << 2) | (o1 << 1) | o2
            base = out_idx & ~mask0 & ~mask1 & ~mask2

            for col in range(8):
                i0 = (col >> 2) & 1
                i1 = (col >> 1) & 1
                i2 = col & 1
                in_idx = base | (i0 << bit0) | (i1 << bit1) | (i2 << bit2)
                expanded[out_idx, in_idx] = op[row, col]

        return expanded

    # ─── Simulation with Noise (Monte Carlo Wave Function) ──────

    def monte_carlo_shot(self, kraus_ops: List[np.ndarray], qubits: Optional[List[int]] = None) -> 'DensityMatrixEngine':
        """
        Genera un singolo shot Monte Carlo per simulazione rumore efficiente.

        Invece di mantenere la matrice densità completa (O(4^n)),
        simula traiettorie pure e media su N shots — più leggero in memoria
        quando servono molti shot.

        Args:
            kraus_ops: Lista di operatori di Kraus (single o multi-qubit)
            qubits: Qubit target (se single-qubit Kraus). Se None, assume full-system operators.
        """
        # Espandi Kraus operators se necessario
        expanded_kraus = []
        for K in kraus_ops:
            if qubits is not None:
                if len(qubits) == 1:
                    K_exp = self._expand_single_qubit_unitary(K, qubits[0])
                elif len(qubits) == 2:
                    K_exp = self._expand_two_qubit_unitary(K, qubits[0], qubits[1])
                elif len(qubits) == 3:
                    K_exp = self._expand_three_qubit_unitary(K, qubits[0], qubits[1], qubits[2])
                else:
                    K_exp = K
            else:
                K_exp = K
            expanded_kraus.append(K_exp)

        # Campiona quale operatore di Kraus applicare
        probs = []
        for K in expanded_kraus:
            # prob = Tr(K ρ K†) = Tr(ρ K†K)
            p = np.real(np.trace(self.density_matrix @ K.conj().T @ K))
            probs.append(max(0, p))

        total_prob = sum(probs)
        if abs(total_prob) < 1e-15:
            return self

        probs = [p / total_prob for p in probs]

        chosen_idx = np.random.choice(len(expanded_kraus), p=probs)
        K_chosen = expanded_kraus[chosen_idx]

        # Applica e normalizza
        new_rho = K_chosen @ self.density_matrix @ K_chosen.conj().T
        trace = np.trace(new_rho).real

        if trace > 1e-15:
            result = DensityMatrixEngine(self.n_qubits)
            result.density_matrix = new_rho / trace
            return result

        return self

    # ─── Visualization & Output ──────────────────────────────────

    def get_probabilities(self) -> np.ndarray:
        """Ottiene le probabilità di misura (diagonale della matrice densità)."""
        diag = np.real(np.diag(self.density_matrix))
        diag = np.clip(diag, 0.0, 1.0)
        total = diag.sum()
        return diag / total if total > 1e-15 else diag

    def to_statevector_approximation(self, threshold: float = 1e-6) -> Optional[np.ndarray]:
        """Se lo stato è quasi puro, approssima con uno statevector."""
        purity = self.get_purity()
        if purity > 0.999:
            # Trova l'autovalore dominante
            eigenvalues, eigenvectors = np.linalg.eigh(self.density_matrix)
            dominant_idx = np.argmax(np.abs(eigenvalues))
            
            if abs(eigenvalues[dominant_idx]) > threshold:
                return eigenvectors[:, dominant_idx] * np.sqrt(abs(eigenvalues[dominant_idx]))
        
        return None

    def bloch_coordinates(self, qubit: int) -> Tuple[float, float, float]:
        """Calcola le coordinate di Bloch per un qubit specifico."""
        # Mantiene il qubit target, traccia fuori tutti gli altri
        rho_reduced = self.partial_trace(
            [qubit],
            [i for i in range(self.n_qubits) if i != qubit],
        )
        
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        x = float(np.real(np.trace(rho_reduced @ X)))
        y = float(np.real(np.trace(rho_reduced @ Y)))
        z = float(np.real(np.trace(rho_reduced @ Z)))
        
        return (x, y, z)

    def partial_trace(self, keep_qubits: List[int], trace_out: List[int]) -> np.ndarray:
        """Calcola la traccia parziale su qubit specifici."""
        n = self.n_qubits
        
        # Dimensione dello spazio ridotto
        kept_dim = 1 << len(keep_qubits)
        reduced_rho = np.zeros((kept_dim, kept_dim), dtype=np.complex128)
        
        for i in range(self.dim):
            for j in range(self.dim):
                # Verifica che i qubit da tracciare abbiano lo stesso valore
                match = True
                for t_q in trace_out:
                    if ((i >> (n - 1 - t_q)) & 1) != ((j >> (n - 1 - t_q)) & 1):
                        match = False
                        break
                
                if match:
                    # Mappa gli indici i, j allo spazio ridotto
                    i_reduced = self._map_to_reduced(i, keep_qubits, n)
                    j_reduced = self._map_to_reduced(j, keep_qubits, n)
                    reduced_rho[i_reduced, j_reduced] += self.density_matrix[i, j]
        
        return reduced_rho

    def _map_to_reduced(self, idx: int, keep_qubits: List[int], total_qubits: int) -> int:
        """Mappa un indice globale allo spazio ridotto."""
        result = 0
        for pos, q in enumerate(keep_qubits):
            bit_pos = total_qubits - 1 - q
            bit = (idx >> bit_pos) & 1
            result |= (bit << (len(keep_qubits) - 1 - pos))
        
        return result

    def __repr__(self):
        purity = self.get_purity() if hasattr(self, '_purity') else self.get_purity()
        return (f"DensityMatrix(n_qubits={self.n_qubits}, "
                f"dim={self.dim}x{self.dim}, "
                f"purity={purity:.4f}, "
                f"gates_applied={self.gate_count})")
