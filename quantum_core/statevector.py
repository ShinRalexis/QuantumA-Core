"""
Statevector Engine — Motore principale di simulazione quantistica pura.

Rappresenta lo stato quantistico come vettore di 2^n ampiezze complesse
(complex128). L'applicazione dei gate è completamente vettorizzata con
`numpy.einsum`: lo statevector viene rimodellato (reshape) in modo da isolare
gli assi dei qubit target e la matrice del gate viene contratta a livello C,
senza loop Python sul vettore di stato. Questo permette di gestire qubit non
adiacenti e in ordine arbitrario.

Limiti pratici su hardware consumer:
- statevector puro: ~28 qubit su CPU, ~27 su GPU (8 GB VRAM)
- la crescita è esponenziale: 2^n × 16 byte per lo stato
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

from quantum_core.gates import QuantumGateLibrary

# Sorgente unica delle matrici dei gate, condivisa con il backend GPU e con il
# resto del simulatore. Evita di duplicare le definizioni (e le convenzioni di
# segno) in più punti: CPU e GPU usano esattamente le stesse matrici.
_GATE_LIB = QuantumGateLibrary()


class StatevectorEngine:
    """
    Motore di simulazione statevector pura.
    
    Rappresenta uno stato quantistico |ψ⟩ come vettore complesso di dimensione 2^n.
    Supporta gate single/two/qubit, misurazione, e ottimizzazioni di caching.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state_dim = 1 << n_qubits  # 2^n
        
        # Statevector: array di ampiezze complesse (float64 + imag)
        self.amplitudes = np.zeros(self.state_dim, dtype=np.complex128)
        
        # Inizializzazione allo stato |00...0⟩
        self.amplitudes[0] = 1.0 + 0j

        # Performance metrics
        self.gate_count = 0
        self.total_simulation_time_ns = 0

    def reset(self):
        """Resetta lo stato a |00...0⟩."""
        self.amplitudes.fill(0)
        self.amplitudes[0] = 1.0 + 0j
        self.gate_count = 0

    @property
    def normalized(self) -> bool:
        """Verifica se lo stato è normalizzato."""
        norm = np.sum(np.abs(self.amplitudes) ** 2)
        return abs(norm - 1.0) < 1e-10

    def normalize(self):
        """Normalizza lo stato quantistico."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-15:
            self.amplitudes /= norm

    # ─── Quantum State Accessors ──────────────────────────────────────

    def get_amplitude(self, computational_basis_state: int) -> complex:
        """Ottieni l'ampiezza di uno stato computazionale specifico."""
        return self.amplitudes[computational_basis_state]

    def probability(self, computational_basis_state: int) -> float:
        """Ottieni la probabilità di uno stato computazionale."""
        return float(np.abs(self.amplitudes[computational_basis_state]) ** 2)

    def get_all_probabilities(self) -> np.ndarray:
        """Ottiene tutte le probabilità di base computazionale."""
        return np.abs(self.amplitudes) ** 2

    def get_bloch_vector(self, target_qubit: int) -> Tuple[float, float, float]:
        """
        Calcola il vettore di Bloch per un qubit specifico.

        Traccia fuori tutti gli altri qubit e calcola ⟨X⟩, ⟨Y⟩, ⟨Z⟩.
        Usa la formula: ⟨σ_k⟩ = Σ_{i,j} ψ*_i (O_k)_{ij} ψ_j
        dove O_k è l'operatore di Pauli sul qubit target.
        """
        n = self.n_qubits
        bit_pos = n - 1 - target_qubit
        mask = 1 << bit_pos

        indices = np.arange(self.state_dim)
        probs = np.abs(self.amplitudes) ** 2

        # ⟨Z⟩ = P(bit=0) - P(bit=1)
        signs = np.where(indices & mask, -1.0, 1.0)
        z_exp = float(np.dot(signs, probs))

        # ⟨X⟩ = 2·Re(Σ ψ*_{bit=0} · ψ_{bit=1}), ⟨Y⟩ = 2·Im(stessa somma)
        i_idx = np.where((indices & mask) == 0)[0]
        j_idx = i_idx | mask
        overlap = np.sum(np.conj(self.amplitudes[i_idx]) * self.amplitudes[j_idx])
        x_exp = float(2.0 * overlap.real)
        y_exp = float(2.0 * overlap.imag)

        return (x_exp, y_exp, z_exp)

    # ─── Measurement ────────────────────────────────────────────────

    def measure(self, target_qubit: int, shots: int = 1, collapse: bool = True) -> List[int]:
        """
        Misura un qubit nella base computazionale.

        Args:
            target_qubit: Indice del qubit da misurare (0 = MSB/first qubit)
            shots: Numero di campioni da restituire
            collapse: Se True, esegue una misura proiettiva che collassa lo stato

        Returns:
            Lista di risultati (0 o 1), lunga `shots`

        Note:
            - Con collapse=True la misura è proiettiva: si campiona una volta e
              lo stato collassa. Misure ripetute dello stesso qubit dopo il
              collasso danno per forza lo stesso esito, quindi viene restituito
              l'esito ripetuto `shots` volte.
            - Con collapse=False si restituiscono `shots` campioni i.i.d. della
              distribuzione marginale, lasciando lo stato invariato.
        """
        if target_qubit < 0 or target_qubit >= self.n_qubits:
            raise ValueError(f"Qubit index out of range: {target_qubit}")

        mask = 1 << (self.n_qubits - 1 - target_qubit)
        indices = np.arange(self.state_dim)
        probs = np.abs(self.amplitudes) ** 2
        prob_1 = float(probs[(indices & mask) != 0].sum())

        if collapse:
            outcome = int(np.random.random() < prob_1)
            self._collapse_state(target_qubit, outcome)
            return [outcome] * shots

        return (np.random.random(shots) < prob_1).astype(int).tolist()

    def _collapse_state(self, target_qubit: int, outcome: int):
        """Collassa lo stato quantistico dopo una misura (vettorizzato)."""
        mask = 1 << (self.n_qubits - 1 - target_qubit)
        bit_is_one = (np.arange(self.state_dim) & mask) != 0
        # Mantieni solo le ampiezze coerenti con l'esito misurato, azzera le altre.
        keep = bit_is_one if outcome == 1 else ~bit_is_one
        self.amplitudes = np.where(keep, self.amplitudes, 0.0 + 0.0j)
        self.normalize()

    def measure_all(self, collapse: bool = True) -> int:
        """Misura tutti i qubit e ritorna l'indice dello stato collassato."""
        probs = np.abs(self.amplitudes) ** 2
        # Normalizza
        total = probs.sum()
        if total > 1e-15:
            probs = probs / total
        
        outcome_idx = int(np.random.choice(range(self.state_dim), p=probs))
        
        if collapse:
            self.amplitudes.fill(0)
            self.amplitudes[outcome_idx] = 1.0 + 0j
        
        return outcome_idx

    # ─── State Preparation ──────────────────────────────────────────

    def prepare_state(self, amplitudes: List[complex]):
        """Prepara uno stato arbitrario."""
        if len(amplitudes) != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} amplitudes, got {len(amplitudes)}")
        
        self.amplitudes = np.array(amplitudes, dtype=np.complex128)
        self.normalize()

    def prepare_bell_state(self, qubit_a: int, qubit_b: int):
        """Prepara uno stato Bell (|00⟩ + |11⟩)/√2 su due qubit."""
        # Applica H al primo qubit e CNOT tra i due
        self.apply_gate('H', qubit_a)
        self.apply_two_qubit_gate('CNOT', qubit_a, qubit_b)

    def prepare_ghz_state(self):
        """Prepara uno stato GHZ (|0...0⟩ + |1...1⟩)/√2."""
        self.apply_gate('H', 0)
        for i in range(1, self.n_qubits):
            self.apply_two_qubit_gate('CNOT', 0, i)

    # ─── Single Qubit Gates ────────────────────────────────────────

    def apply_gate(self, gate_name: str, target_qubit: int, params: Optional[Dict] = None):
        """Applica un gate single-qubit al qubit specificato.

        Le matrici provengono dalla QuantumGateLibrary condivisa (_GATE_LIB),
        unica fonte di verità per CPU e GPU.
        """
        matrix = _GATE_LIB.get_matrix(gate_name, **(params or {}))
        self._apply_matrix(matrix, [target_qubit])
        self.gate_count += 1

    def apply_custom_gate(self, matrix: np.ndarray, target_qubits: List[int]):
        """Applica un gate custom (matrice unitaria) ai qubit specificati."""
        if len(target_qubits) == 1:
            if matrix.shape != (2, 2):
                raise ValueError("Single-qubit gate must be 2x2")
            self._apply_matrix(matrix, target_qubits)
        elif len(target_qubits) == 2:
            if matrix.shape != (4, 4):
                raise ValueError("Two-qubit gate must be 4x4")
            self._apply_matrix_2q(matrix, target_qubits[0], target_qubits[1])
        elif len(target_qubits) == 3:
            if matrix.shape != (8, 8):
                raise ValueError("Three-qubit gate must be 8x8")
            self._apply_matrix_3q(matrix, target_qubits)
        else:
            raise ValueError(f"Custom gates support max 3 qubits, got {len(target_qubits)}")

    # ─── Two Qubit Gates ──────────────────────────────────────────

    def apply_two_qubit_gate(self, gate_name: str, control_qubit: int, target_qubit: int, params: Optional[Dict] = None):
        """Applica un gate two-qubit.

        Le matrici provengono dalla QuantumGateLibrary condivisa (_GATE_LIB).
        """
        matrix = _GATE_LIB.get_two_qubit_gate(gate_name, **(params or {}))
        self._apply_matrix_2q(matrix, control_qubit, target_qubit)
        self.gate_count += 1

    # ─── Core Matrix Application (Vectorized NumPy) ─────────────────

    def _apply_matrix(self, matrix: np.ndarray, qubits: List[int]):
        """
        Applica una matrice 2x2 su un singolo qubit.

        Vettorizzata: reshape dello statevector in 3 assi (isolando il qubit
        target) + contrazione einsum a livello C, senza loop Python.
        """
        target = qubits[0]
        # Reshape: (2^target, 2, 2^(n-1-target))
        # Asse 1 = il qubit target
        A = self.amplitudes.reshape(1 << target, 2, 1 << (self.n_qubits - 1 - target))
        # new[i,a,j] = sum_b  M[a,b] * A[i,b,j]
        A_new = np.einsum('ab,ibj->iaj', matrix, A, optimize=True)
        self.amplitudes = A_new.reshape(self.state_dim)

    def _apply_matrix_2q(self, matrix: np.ndarray, qubit_a: int, qubit_b: int):
        """
        Applica una matrice 4x4 su due qubit.

        Versione fully-vectorized: reshape in 5 assi + einsum C-level.
        Funziona per qubit non adiacenti e in qualsiasi ordine.
        """
        if qubit_a == qubit_b:
            raise ValueError("Two-qubit gate requires distinct qubits")

        n = self.n_qubits
        q_min = min(qubit_a, qubit_b)
        q_max = max(qubit_a, qubit_b)

        d1 = 1 << q_min                   # qubit 0..q_min-1
        d3 = 1 << (q_max - q_min - 1)     # qubit q_min+1..q_max-1
        d5 = 1 << (n - 1 - q_max)         # qubit q_max+1..n-1

        M = matrix.reshape(2, 2, 2, 2)    # (out_a, out_b, in_a, in_b)
        A = self.amplitudes.reshape(d1, 2, d3, 2, d5)

        if qubit_a < qubit_b:
            # asse 1 = qubit_a (q_min), asse 3 = qubit_b (q_max)
            # result[l,oa,m,ob,r] = sum_{ia,ib} M[oa,ob,ia,ib] * A[l,ia,m,ib,r]
            A_new = np.einsum('abcd,lcmdo->lambo', M, A, optimize=True)
        else:
            # asse 1 = qubit_b (q_min), asse 3 = qubit_a (q_max)
            # result[l,ob,m,oa,r] = sum_{ia,ib} M[oa,ob,ia,ib] * A[l,ib,m,ia,r]
            A_new = np.einsum('abcd,ldmco->lbmao', M, A, optimize=True)

        self.amplitudes = A_new.reshape(self.state_dim)
    def _apply_matrix_3q(self, matrix: np.ndarray, qubits: List[int]):
        """
        Applica una matrice 8x8 su tre qubit (Toffoli e simili).

        Vettorizzata con tensordot: lo statevector viene visto come tensore di
        n assi da 2 (reshape([2]*n)), dove l'asse i corrisponde al qubit i
        (convenzione big-endian). La matrice 8x8 è rimodellata in (2,2,2,2,2,2)
        con assi (out_q0,out_q1,out_q2, in_q0,in_q1,in_q2) e i suoi assi di
        input vengono contratti con gli assi (q0,q1,q2) dello stato. Gli assi
        di output vengono poi riportati nelle posizioni dei qubit corretti.
        """
        if len(qubits) != 3:
            raise ValueError("_apply_matrix_3q requires exactly 3 qubits")
        n = self.n_qubits
        q0, q1, q2 = qubits

        M = matrix.reshape(2, 2, 2, 2, 2, 2)  # (o0,o1,o2, i0,i1,i2)
        T = self.amplitudes.reshape([2] * n)

        # Contrae gli assi di input (3,4,5) di M con gli assi (q0,q1,q2) di T.
        R = np.tensordot(M, T, axes=([3, 4, 5], [q0, q1, q2]))

        # R ha gli assi: [q0, q1, q2] + (altri qubit in ordine crescente).
        others = [q for q in range(n) if q not in (q0, q1, q2)]
        r_axis_qubit = [q0, q1, q2] + others
        perm = [r_axis_qubit.index(i) for i in range(n)]

        self.amplitudes = np.transpose(R, perm).reshape(self.state_dim)

    # ─── Fidelity & Metrics ────────────────────────────────────────

    def fidelity(self, other_statevector: np.ndarray) -> float:
        """Calcola la fedeltà tra due stati quantistici."""
        overlap = np.abs(np.vdot(self.amplitudes, other_statevector)) ** 2
        return float(overlap.real)

    def entropy(self) -> float:
        """Calcola l'entropia di von Neumann (misura di entanglement)."""
        probs = np.abs(self.amplitudes) ** 2
        # Filtra probabilità zero per evitare log(0)
        nonzero_probs = probs[probs > 1e-30]
        return float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    def get_state_dict(self) -> Dict[str, complex]:
        """Ritorna lo stato come dizionario {bitstring: amplitude}."""
        result = {}
        for i in range(self.state_dim):
            if abs(self.amplitudes[i]) > 1e-6:
                bitstr = format(i, f'0{self.n_qubits}b')
                result[bitstr] = complex(self.amplitudes[i])
        return result

    def __repr__(self):
        non_zero = sum(1 for a in self.amplitudes if abs(a) > 1e-6)
        return (f"Statevector(n_qubits={self.n_qubits}, "
                f"dim={self.state_dim}, "
                f"non_zero_amplitudes={non_zero}, "
                f"gates_applied={self.gate_count})")
