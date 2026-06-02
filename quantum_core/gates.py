"""
Quantum Gate Library — Libreria completa di gate quantistici.

Include:
- Single qubit gates (Pauli, Clifford, T, rotazioni)
- Two-qubit gates (CNOT, CZ, SWAP, iSWAP, RZZ, CRZ)
- Three-qubit gates (CCNOT/Toffoli, CCZ)
- Utility di gate fusion (composizione di gate consecutivi in un'unica matrice)

Tutte le matrici sono in convenzione big-endian (qubit 0 = MSB) e verificate
unitarie nella suite di test.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GateDefinition:
    """Definizione di un gate quantistico."""
    name: str
    matrix: np.ndarray  # Matrice unitaria del gate
    n_qubits: int       # Numero di qubit target
    is_diagonal: bool   # Se il gate è diagonale (per ottimizzazione)


class QuantumGateLibrary:
    """
    Libreria completa di gate quantistici standard.
    
    Tutti i gate sono definiti nella base computazionale.
    Supporta gate fusion per ottimizzazione del circuito.
    """

    def __init__(self):
        self._gates: Dict[str, GateDefinition] = {}
        self._build_standard_gates()

    def _build_standard_gates(self):
        """Costruisce tutti i gate standard."""
        
        # ─── Single Qubit Gates ──────────────────────────────────────
        
        self.register(GateDefinition(
            'I', np.eye(2, dtype=np.complex128), 1, True
        ))
        
        self.register(GateDefinition(
            'X', np.array([[0, 1], [1, 0]], dtype=np.complex128), 1, False
        ))
        
        self.register(GateDefinition(
            'Y', np.array([[0, -1j], [1j, 0]], dtype=np.complex128), 1, False
        ))
        
        self.register(GateDefinition(
            'Z', np.array([[1, 0], [0, -1]], dtype=np.complex128), 1, True
        ))
        
        # Hadamard
        self.register(GateDefinition(
            'H', 
            np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2), 
            1, False
        ))
        
        # Phase gate S = √Z
        self.register(GateDefinition(
            'S', 
            np.array([[1, 0], [0, 1j]], dtype=np.complex128), 
            1, True
        ))
        
        # T gate (π/8)
        self.register(GateDefinition(
            'T', 
            np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128), 
            1, True
        ))
        
        # ─── Rotation Gates ──────────────────────────────────────────
        
        self.register_rotation('RX', 'x')
        self.register_rotation('RY', 'y')
        self.register_rotation('RZ', 'z')

    def register(self, gate_def: GateDefinition):
        """Registra un nuovo gate."""
        self._gates[gate_def.name] = gate_def

    def register_rotation(self, name: str, axis: str):
        """Registra un gate di rotazione parametrizzato."""
        
        def make_rx(theta):
            c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
            return np.array([[c, s], [s, c]], dtype=np.complex128)
        
        def make_ry(theta):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        
        def make_rz(phi):
            return np.array([
                [np.exp(-1j * phi / 2), 0],
                [0, np.exp(1j * phi / 2)]
            ], dtype=np.complex128)
        
        # Register con parametri default (π/2)
        if axis == 'x':
            self._gates[f'{name}'] = GateDefinition(
                name, make_rx(np.pi / 2), 1, False
            )
        elif axis == 'y':
            self._gates[f'{name}'] = GateDefinition(
                name, make_ry(np.pi / 2), 1, False
            )
        else:
            self._gates[f'{name}'] = GateDefinition(
                name, make_rz(np.pi / 2), 1, True
            )

    def get_matrix(self, gate_name: str, **params) -> np.ndarray:
        """Ottiene la matrice di un gate, supportando parametri per rotazioni."""
        if gate_name in ('RX', 'RY', 'RZ'):
            theta = params.get('theta', params.get('phi', np.pi / 2))
            if gate_name == 'RX':
                c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
                return np.array([[c, s], [s, c]], dtype=np.complex128)
            elif gate_name == 'RY':
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                return np.array([[c, -s], [s, c]], dtype=np.complex128)
            else: # RZ
                return np.array([
                    [np.exp(-1j * theta / 2), 0],
                    [0, np.exp(1j * theta / 2)]
                ], dtype=np.complex128)
        
        if gate_name not in self._gates:
            raise ValueError(f"Unknown gate: {gate_name}")
        return self._gates[gate_name].matrix

    @property
    def available_gates(self) -> List[str]:
        """Lista dei gate disponibili."""
        return list(self._gates.keys())

    # ─── Two-Qubit Gates ────────────────────────────────────────

    @staticmethod
    def cnot_matrix() -> np.ndarray:
        """Matrice CNOT nella base computazionale."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)

    @staticmethod
    def cz_matrix() -> np.ndarray:
        """Matrice CZ."""
        return np.diag([1, 1, 1, -1]).astype(np.complex128)

    @staticmethod
    def swap_matrix() -> np.ndarray:
        """Matrice SWAP."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)

    @staticmethod
    def iswap_matrix() -> np.ndarray:
        """Matrice iSWAP."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)

    @staticmethod
    def rzz_matrix(theta: float) -> np.ndarray:
        """Matrice RZZ(θ) = exp(-i θ/2 Z⊗Z): diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2})."""
        return np.diag([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2),
        ]).astype(np.complex128)

    @staticmethod
    def crz_matrix(phi: float) -> np.ndarray:
        """Matrice CRZ(φ): applica RZ(φ) al target quando control=1."""
        return np.diag([1, 1, np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]).astype(np.complex128)

    @staticmethod
    def ccnot_matrix() -> np.ndarray:
        """Matrice CCNOT (Toffoli) 8×8."""
        matrix = np.eye(8, dtype=np.complex128)
        # Swap |110⟩ ↔ |111⟩
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        return matrix

    @staticmethod
    def ccz_matrix() -> np.ndarray:
        """Matrice CCZ (Controlled-Controlled-Z) 8×8."""
        matrix = np.eye(8, dtype=np.complex128)
        matrix[7, 7] = -1  # Solo |111⟩ riceve fase π
        return matrix

    def get_two_qubit_gate(self, name: str, **params) -> np.ndarray:
        """Ottiene una matrice two-qubit."""
        if name == 'CNOT':
            return self.cnot_matrix()
        elif name == 'CZ':
            return self.cz_matrix()
        elif name == 'SWAP':
            return self.swap_matrix()
        elif name == 'iSWAP':
            return self.iswap_matrix()
        elif name == 'RZZ':
            theta = params.get('theta', np.pi / 4)
            return self.rzz_matrix(theta)
        elif name == 'CRZ':
            phi = params.get('phi', np.pi / 4)
            return self.crz_matrix(phi)
        elif name == 'CPHASE':
            theta = params.get('theta', np.pi / 4)
            return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(np.complex128)
        elif name == 'CCNOT' or name == 'Toffoli':
            return self.ccnot_matrix()
        else:
            raise ValueError(f"Unknown two-qubit gate: {name}")

    # ─── Gate Fusion (composizione di gate consecutivi) ──────────

    @staticmethod
    def fuse_single_gates(gate_a: np.ndarray, gate_b: np.ndarray) -> np.ndarray:
        """
        Fuse due gate single-qubit consecutivi.
        
        Se applichi prima A poi B al qubit target, l'effetto combinato è BA.
        
        Speedup: 2x (una moltiplicazione matrice-vettore invece di due)
        """
        return gate_b @ gate_a

    def fuse_diagonal_gates(self, gates: List[Tuple[str, Dict]]) -> np.ndarray:
        """
        Fuse gate single-qubit diagonali consecutivi in un'unica matrice 2x2.

        Tutti i gate diagonali agiscono come diag([α, β]) con autovalori
        distinti su |0⟩ e |1⟩:
            I       → [1, 1]
            Z       → [1, -1]
            S       → [1, i]
            T       → [1, exp(iπ/4)]
            RZ(φ)   → [exp(-iφ/2), exp(iφ/2)]
        """
        alpha = 1.0 + 0j  # autovalore su |0⟩
        beta = 1.0 + 0j   # autovalore su |1⟩

        for gate_name, params in gates:
            if gate_name == 'I':
                continue
            elif gate_name == 'Z':
                beta *= -1
            elif gate_name == 'S':
                beta *= 1j
            elif gate_name == 'T':
                beta *= np.exp(1j * np.pi / 4)
            elif gate_name == 'RZ':
                phi = (params or {}).get('phi', 0)
                alpha *= np.exp(-1j * phi / 2)
                beta *= np.exp(1j * phi / 2)
            else:
                raise ValueError(
                    f"fuse_diagonal_gates: '{gate_name}' non è un gate "
                    f"single-qubit diagonale supportato"
                )

        return np.diag([alpha, beta]).astype(np.complex128)

    def optimize_circuit(self, circuit: List[dict]) -> List[dict]:
        """
        Ottimizza un circuito applicando gate fusion.
        
        Args:
            circuit: Lista di {'gate': nome, 'qubits': [target], 'params': {}}
        
        Returns:
            Circuito ottimizzato con gate fusion applicata
        """
        if not circuit:
            return []
        
        optimized = [circuit[0].copy()]
        
        for i in range(1, len(circuit)):
            prev = optimized[-1]
            curr = circuit[i].copy()
            
            # Fuse solo se stesso qubit target e gate consecutivi
            if (prev['qubits'] == curr['qubits'] and 
                prev.get('params') is not None):
                
                mat_a = self._gate_to_matrix(prev)
                mat_b = self._gate_to_matrix(curr)
                
                if mat_a.shape[0] == 2:  # Single-qubit fusion
                    fused = self.fuse_single_gates(mat_a, mat_b)
                    
                    # Approssima il gate fuso al nome più vicino
                    approx_name = self._approximate_gate(fused)
                    if approx_name:
                        optimized[-1] = {
                            'gate': approx_name,
                            'qubits': prev['qubits'],
                            'params': {}
                        }
                    else:
                        optimized[-1] = {
                            'gate': f'CUSTOM({approx_name or "FUSED"})',
                            'qubits': prev['qubits'],
                            'matrix': fused,
                            'params': {'fused_from': [prev['gate'], curr['gate']]}
                        }
                    continue
            
            optimized.append(curr)
        
        return optimized

    def _gate_to_matrix(self, gate_def: dict) -> np.ndarray:
        """Converte una definizione di gate in matrice."""
        name = gate_def['gate']
        
        if 'matrix' in gate_def:
            return gate_def['matrix']
        
        try:
            return self.get_two_qubit_gate(name, **gate_def.get('params', {}))
        except ValueError:
            pass
        
        # Single qubit gates
        single_gates = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]]),
            'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            'S': np.array([[1, 0], [0, 1j]]),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
        }
        
        if name in single_gates:
            return single_gates[name]
        
        raise ValueError(f"Cannot convert gate to matrix: {name}")

    def _approximate_gate(self, matrix: np.ndarray) -> Optional[str]:
        """Approssima una matrice unitaria al nome di gate standard più vicino."""
        # Normalizza la fase globale
        phase = np.angle(matrix[0, 0]) if abs(matrix[0, 0]) > 1e-6 else 0
        normalized = matrix * np.exp(-1j * phase)
        
        candidates = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]]),
            'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        }
        
        best_name = None
        best_fidelity = 0
        
        for name, std_matrix in candidates.items():
            fidelity = abs(np.trace(normalized @ std_matrix.conj().T)) ** 2 / 4
            if fidelity > best_fidelity and fidelity > 0.95:
                best_fidelity = fidelity
                best_name = name
        
        return best_name

    # ─── Common Circuit Patterns ────────────────────────────────

    @staticmethod
    def bell_state_circuit() -> List[dict]:
        """Circuito per creare uno stato Bell."""
        return [
            {'gate': 'H', 'qubits': [0], 'params': {}},
            {'gate': 'CNOT', 'qubits': [0, 1], 'params': {}}
        ]

    @staticmethod
    def ghz_state_circuit(n_qubits: int) -> List[dict]:
        """Circuito per creare uno stato GHZ."""
        circuit = [{'gate': 'H', 'qubits': [0], 'params': {}}]
        for i in range(1, n_qubits):
            circuit.append({'gate': 'CNOT', 'qubits': [0, i], 'params': {}})
        return circuit

    @staticmethod
    def grover_iteration(n_qubits: int) -> List[dict]:
        """Una iterazione di Grover (oracle + diffusion)."""
        circuit = []
        
        # Oracle: applica Z a tutti |11...1⟩
        for i in range(1, n_qubits):
            circuit.append({'gate': 'H', 'qubits': [i], 'params': {}})
        circuit.append({'gate': 'CCNOT', 'qubits': list(range(n_qubits)), 'params': {}})
        for i in range(1, n_qubits):
            circuit.append({'gate': 'H', 'qubits': [i], 'params': {}})
        
        # Diffusion operator: H^⊗n · 2|0⟩⟨0| - I · H^⊗n
        for i in range(n_qubits):
            circuit.append({'gate': 'H', 'qubits': [i], 'params': {}})
        
        for i in range(n_qubits):
            circuit.append({'gate': 'X', 'qubits': [i], 'params': {}})
        
        # Multi-controlled Z (Toffoli chain)
        for i in range(1, n_qubits):
            circuit.append({'gate': 'CNOT', 'qubits': [0, i], 'params': {}})
        circuit.append({'gate': 'Z', 'qubits': [n_qubits - 1], 'params': {}})
        for i in range(n_qubits - 1, 0, -1):
            circuit.append({'gate': 'CNOT', 'qubits': [0, i], 'params': {}})
        
        for i in range(n_qubits):
            circuit.append({'gate': 'X', 'qubits': [i], 'params': {}})
        
        for i in range(n_qubits):
            circuit.append({'gate': 'H', 'qubits': [i], 'params': {}})
        
        return circuit
