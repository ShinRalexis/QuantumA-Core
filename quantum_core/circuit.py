"""
Quantum Circuit Builder — Costruttore di circuiti quantistici.

DSL (Domain Specific Language) per definire circuiti quantistici con:
- Gate definition con parametri
- Timing e scheduling
- Validazione del circuito
- Ottimizzazione automatica di base (gate cancellation, merge di RZ)

API in stile builder ispirata a Google Cirq / IBM Qiskit.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from quantum_core.gates import QuantumGateLibrary


@dataclass
class CircuitInstruction:
    """Istruzione di un circuito quantistico."""
    gate_name: str
    qubits: List[int]
    params: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0       # Durata in μs (per scheduling)
    label: Optional[str] = None  # Etichetta umana


@dataclass 
class QuantumCircuit:
    """
    Rappresentazione di un circuito quantistico.
    
    Supporta:
    - Definizione esplicita di gate
    - Sub-circuit composition
    - Timing analysis
    - Depth e width calculation
    - Ottimizzazione automatica
    """
    
    n_qubits: int
    instructions: List[CircuitInstruction] = field(default_factory=list)
    name: str = "unnamed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")

    # ─── Circuit Building API ──────────────────────────────────────

    def h(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica Hadamard."""
        self._add_instruction('H', [qubit], {}, label=label)
        return self

    def x(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica Pauli-X (NOT)."""
        self._add_instruction('X', [qubit], {}, label=label)
        return self

    def y(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica Pauli-Y."""
        self._add_instruction('Y', [qubit], {}, label=label)
        return self

    def z(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica Pauli-Z."""
        self._add_instruction('Z', [qubit], {}, label=label)
        return self

    def s(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica Phase gate (S)."""
        self._add_instruction('S', [qubit], {}, label=label)
        return self

    def t(self, qubit: int, label: str = None) -> 'QuantumCircuit':
        """Applica T gate."""
        self._add_instruction('T', [qubit], {}, label=label)
        return self

    def rx(self, qubit: int, theta: float = np.pi / 2, label: str = None) -> 'QuantumCircuit':
        """Rotazione RX(θ)."""
        self._add_instruction('RX', [qubit], {'theta': theta}, label=label)
        return self

    def ry(self, qubit: int, theta: float = np.pi / 2, label: str = None) -> 'QuantumCircuit':
        """Rotazione RY(θ)."""
        self._add_instruction('RY', [qubit], {'theta': theta}, label=label)
        return self

    def rz(self, qubit: int, phi: float = np.pi / 2, label: str = None) -> 'QuantumCircuit':
        """Rotazione RZ(φ)."""
        self._add_instruction('RZ', [qubit], {'phi': phi}, label=label)
        return self

    def cx(self, control: int, target: int, label: str = None) -> 'QuantumCircuit':
        """Applica CNOT."""
        self._add_instruction('CNOT', [control, target], {}, label=label)
        return self

    def cz(self, qubit_a: int, qubit_b: int, label: str = None) -> 'QuantumCircuit':
        """Applica CZ."""
        self._add_instruction('CZ', [qubit_a, qubit_b], {}, label=label)
        return self

    def swap(self, qubit_a: int, qubit_b: int, label: str = None) -> 'QuantumCircuit':
        """Applica SWAP."""
        self._add_instruction('SWAP', [qubit_a, qubit_b], {}, label=label)
        return self

    def ccnot(self, control1: int, control2: int, target: int, label: str = None) -> 'QuantumCircuit':
        """Applica gate Toffoli (CCNOT) — doppio controlled-NOT."""
        self._add_instruction('CCNOT', [control1, control2, target], {}, label=label)
        return self

    def iswap(self, qubit_a: int, qubit_b: int, label: str = None) -> 'QuantumCircuit':
        """Applica iSWAP."""
        self._add_instruction('iSWAP', [qubit_a, qubit_b], {}, label=label)
        return self

    def crz(self, control: int, target: int, phi: float = np.pi / 4,
            label: str = None) -> 'QuantumCircuit':
        """Applica CRZ(φ) — RZ(φ) condizionale."""
        self._add_instruction('CRZ', [control, target], {'phi': phi}, label=label)
        return self

    def rzz(self, qubit_a: int, qubit_b: int, theta: float = np.pi / 4,
            label: str = None) -> 'QuantumCircuit':
        """Applica RZZ(θ)."""
        self._add_instruction('RZZ', [qubit_a, qubit_b], {'theta': theta}, label=label)
        return self

    def cphase(self, control: int, target: int, theta: float = np.pi / 4, 
               label: str = None) -> 'QuantumCircuit':
        """Applica Controlled-Phase(θ)."""
        self._add_instruction('CPHASE', [control, target], {'theta': theta}, label=label)
        return self

    def measure(self, qubits: List[int], register_name: str = "meas") -> 'QuantumCircuit':
        """Aggiunge operazioni di misura."""
        for q in qubits:
            self._add_instruction('MEASURE', [q], {'register': register_name})
        return self

    def barrier(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """Aggiunge un barrier (per scheduling)."""
        self.instructions.append(CircuitInstruction(
            gate_name='BARRIER',
            qubits=qubits or list(range(self.n_qubits)),
            params={}
        ))
        return self

    def subcircuit(self, circuit: 'QuantumCircuit', 
                   qubit_mapping: Dict[int, int]) -> 'QuantumCircuit':
        """Inserisce un sotto-circuito con mappatura dei qubit."""
        for inst in circuit.instructions:
            mapped_qubits = [qubit_mapping.get(q, q) for q in inst.qubits]
            
            new_inst = CircuitInstruction(
                gate_name=inst.gate_name,
                qubits=mapped_qubits,
                params=dict(inst.params),
                duration=inst.duration,
                label=f"{circuit.name}/{inst.label}" if inst.label else circuit.name
            )
            self.instructions.append(new_inst)
        
        return self

    def _add_instruction(self, gate_name: str, qubits: List[int], 
                         params: Dict[str, float], label: str = None):
        """Aggiunge un'istruzione al circuito."""
        # Validazione
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.n_qubits})")
        
        if len(qubits) > 3:
            raise ValueError(f"Max 3-qubit gates supported, got {len(qubits)}-qubit gate")
        
        # Assegna durata default
        duration = 0.01 if len(qubits) == 1 else 0.05
        
        self.instructions.append(CircuitInstruction(
            gate_name=gate_name,
            qubits=qubits,
            params=params,
            duration=duration,
            label=label
        ))

    # ─── Circuit Analysis ──────────────────────────────────────

    @property
    def depth(self) -> int:
        """Calcola la profondità del circuito (numero di layer seriali)."""
        if not self.instructions:
            return 0
        
        # Timing-based depth calculation
        qubit_times = {q: 0.0 for q in range(self.n_qubits)}
        
        for inst in self.instructions:
            if inst.gate_name == 'BARRIER':
                max_time = max(qubit_times.values())
                for q in qubit_times:
                    qubit_times[q] = max_time
            
            elif inst.gate_name != 'MEASURE':
                max_start = max(qubit_times.get(q, 0) for q in inst.qubits)
                end_time = max_start + inst.duration
                
                for q in inst.qubits:
                    qubit_times[q] = end_time
        
        return int(np.ceil(max(qubit_times.values()) / 0.01)) if qubit_times else 0

    @property
    def width(self) -> int:
        """Numero di qubit del circuito."""
        return self.n_qubits

    @property
    def gate_count(self) -> Dict[str, int]:
        """Conta i gate per tipo."""
        counts = {}
        for inst in self.instructions:
            if inst.gate_name == 'BARRIER':
                continue
            counts[inst.gate_name] = counts.get(inst.gate_name, 0) + 1
        return counts

    @property
    def total_gates(self) -> int:
        """Numero totale di gate."""
        return sum(1 for inst in self.instructions if inst.gate_name != 'BARRIER')

    @property
    def two_qubit_gates(self) -> int:
        """Numero di gate a due qubit (metrica chiave per NISQ)."""
        return sum(1 for inst in self.instructions 
                   if len(inst.qubits) == 2 and inst.gate_name != 'BARRIER')

    def get_timing_analysis(self) -> Dict[str, Any]:
        """Analisi dettagliata del timing."""
        qubit_times = {q: [] for q in range(self.n_qubits)}
        
        current_time = 0.0
        for inst in self.instructions:
            if inst.gate_name == 'BARRIER':
                continue
            
            start_time = max(current_time, 
                           max((t[-1] if t else 0) for t in qubit_times.values()))
            
            end_time = start_time + inst.duration
            
            for q in inst.qubits:
                qubit_times[q].append((start_time, end_time))
            
            current_time = end_time
        
        total_duration = max(
            (max(t[1] for t in times) if times else 0)
            for times in qubit_times.values()
        )
        
        return {
            'total_duration_us': total_duration,
            'qubit_utilization': {
                q: len(times) > 0 for q, times in qubit_times.items()
            },
            'max_parallel_gates': self._estimate_max_parallel(),
        }

    def _estimate_max_parallel(self) -> int:
        """Stima il massimo parallelismo."""
        # Semplice: conta quanti gate non-overlapping possono essere eseguiti insieme
        if not self.instructions:
            return 0
        
        active = set()
        max_active = 0
        
        for inst in self.instructions:
            qubits_used = set(inst.qubits)
            
            # Rimuovi i qubit già liberi
            active = {q for q in active if q not in qubits_used}
            active.update(qubits_used)
            
            max_active = max(max_active, len(active))
        
        return max_active

    # ─── Optimization ──────────────────────────────────────

    def optimize(self) -> 'QuantumCircuit':
        """Applica ottimizzazioni automatiche."""
        optimized_gates = []
        
        for inst in self.instructions:
            if inst.gate_name == 'BARRIER':
                continue
            
            # Cancellazione gate inversi (X·X = I, H·H = I, ecc.)
            if len(optimized_gates) >= 1 and len(inst.qubits) == 1:
                prev = optimized_gates[-1]

                if (prev.gate_name == inst.gate_name and
                    prev.qubits == inst.qubits and
                    prev.gate_name in ('X', 'H', 'Y', 'Z')):

                    # X·X = I, H·H = I, ecc. — rimuovi il prev e salta il corrente
                    optimized_gates.pop()
                    continue
            
            # S·S = Z, T·T = S
            if (inst.gate_name in ('S', 'T') and 
                optimized_gates and 
                optimized_gates[-1].gate_name == inst.gate_name and
                optimized_gates[-1].qubits == inst.qubits):
                
                if inst.gate_name == 'S':
                    optimized_gates[-1] = CircuitInstruction('Z', inst.qubits, {}, 0.01)
                elif inst.gate_name == 'T':
                    optimized_gates[-1] = CircuitInstruction('S', inst.qubits, {}, 0.01)
                continue
            
            # Merge RZ gates: RZ(φ₁)·RZ(φ₂) = RZ(φ₁+φ₂)
            if (inst.gate_name == 'RZ' and 
                optimized_gates and 
                optimized_gates[-1].gate_name == 'RZ' and
                optimized_gates[-1].qubits == inst.qubits):
                
                merged_phi = optimized_gates[-1].params.get('phi', 0) + inst.params.get('phi', 0)
                # Normalizza a [-π, π]
                merged_phi = ((merged_phi + np.pi) % (2 * np.pi)) - np.pi
                
                if abs(merged_phi) < 1e-10:
                    optimized_gates.pop()  # RZ(0) = I
                else:
                    optimized_gates[-1] = CircuitInstruction(
                        'RZ', inst.qubits, {'phi': merged_phi}, 0.01
                    )
                continue
            
            optimized_gates.append(inst)
        
        # Crea nuovo circuito ottimizzato
        new_circuit = QuantumCircuit(self.n_qubits)
        new_circuit.name = f"{self.name}_opt"
        new_circuit.instructions = optimized_gates
        
        return new_circuit

    # ─── Visualization ──────────────────────────────────────

    def to_ascii(self, max_width: int = 80) -> str:
        """Rappresentazione ASCII del circuito."""
        if not self.instructions:
            return "Circuito vuoto"

        # Greedy left-pack: assegna a ogni istruzione la prima colonna
        # in cui tutti i suoi qubit sono liberi.
        qubit_free = [0] * self.n_qubits
        inst_cols = []
        for inst in self.instructions:
            if inst.gate_name == 'BARRIER':
                col = max(qubit_free)
                for q in range(self.n_qubits):
                    qubit_free[q] = col + 1
            else:
                col = max(qubit_free[q] for q in inst.qubits)
                for q in inst.qubits:
                    qubit_free[q] = col + 1
            inst_cols.append(col)

        n_cols = max(qubit_free)
        if n_cols == 0:
            return "Circuito vuoto"

        # Griglia simboli: grid[qubit][col] = (symbol_str, boxed:bool)
        col_w = [1] * n_cols
        grid = [[None] * n_cols for _ in range(self.n_qubits)]

        for inst, col in zip(self.instructions, inst_cols):
            if inst.gate_name == 'BARRIER':
                for q in range(self.n_qubits):
                    grid[q][col] = ('|', False)
            elif len(inst.qubits) == 1:
                q = inst.qubits[0]
                label = inst.gate_name[:5]
                grid[q][col] = (label, True)
                col_w[col] = max(col_w[col], len(label))
            elif len(inst.qubits) >= 2:
                q0 = inst.qubits[0]
                label = inst.gate_name[:3]
                grid[q0][col] = (label, True)
                col_w[col] = max(col_w[col], len(label))
                # Mark remaining qubits and add vertical connectors
                for k in range(1, len(inst.qubits)):
                    qk = inst.qubits[k]
                    grid[qk][col] = ('*', True)
                # Connettori verticali per qubit intermedi tra primo e ultimo
                q_min = min(inst.qubits)
                q_max = max(inst.qubits)
                for q in range(q_min + 1, q_max):
                    if grid[q][col] is None:
                        grid[q][col] = ('|', False)

        # Render: una riga continua per qubit
        pfx = len(f"q{self.n_qubits - 1}: ")
        lines = []
        for q in range(self.n_qubits):
            parts = [f"q{q}: ".ljust(pfx)]
            for col in range(n_cols):
                w = col_w[col]
                cell = grid[q][col]
                if cell is None:
                    parts.append('-' * (w + 4))
                else:
                    sym, boxed = cell
                    s = sym.center(w)
                    parts.append(f'-[{s}]-' if boxed else f'--{s}--')
            parts.append('-')
            line = ''.join(parts)
            lines.append(line[:max_width] if len(line) > max_width else line)

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Converte il circuito in formato JSON-serializable."""
        return {
            'name': self.name,
            'n_qubits': self.n_qubits,
            'instructions': [
                {
                    'gate': inst.gate_name,
                    'qubits': inst.qubits,
                    'params': inst.params,
                    'duration': inst.duration,
                    'label': inst.label,
                }
                for inst in self.instructions
            ]
        }

    @staticmethod
    def from_dict(data: Dict) -> 'QuantumCircuit':
        """Crea un circuito da formato dict/JSON."""
        circuit = QuantumCircuit(
            n_qubits=data['n_qubits'],
            name=data.get('name', 'imported')
        )
        
        for inst_data in data['instructions']:
            circuit.instructions.append(CircuitInstruction(
                gate_name=inst_data['gate'],
                qubits=inst_data['qubits'],
                params=inst_data.get('params', {}),
                duration=inst_data.get('duration', 0.01),
                label=inst_data.get('label'),
            ))
        
        return circuit

    def __repr__(self):
        return (f"QuantumCircuit(name='{self.name}', "
                f"n_qubits={self.n_qubits}, "
                f"gates={self.total_gates}, "
                f"depth={self.depth})")
