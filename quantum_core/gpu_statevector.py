"""
GPU Statevector Engine — backend opzionale CUDA per QuantumA Core.

Obiettivo:
- usare VRAM/CUDA per lo statevector puro
- mantenere fallback CPU
- supportare gate single-qubit e two-qubit standard
- preservare compatibilità architetturale con il core esistente
"""

from __future__ import annotations

import math
from typing import Optional, List
import numpy as np

try:
    import torch
except Exception:
    torch = None

# Dimensione target di un blocco temporaneo durante l'applicazione di un gate.
# Applicando l'einsum a blocchi sull'asse spettatore più grande, il picco di
# VRAM scende da ~2× la dimensione dello stato a ~(1 + blocco/stato)×. Con
# blocchi da 256 MB, uno stato da 4.29 GB (28 qubit) ha un picco di ~4.55 GB
# invece di 8.58 GB, quindi entra negli 8 GB di VRAM di una GPU consumer.
_CHUNK_TARGET_BYTES = 256 * 1024 * 1024


class GPUStatevectorEngine:
    """Backend GPU per simulazione statevector."""

    def __init__(self, n_qubits: int, dtype=None, device: Optional[str] = None):
        if torch is None:
            raise RuntimeError("PyTorch non disponibile: backend GPU non attivabile")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile")

        self.n_qubits = n_qubits
        self.state_dim = 1 << n_qubits
        self.device = torch.device(device or "cuda")
        self.dtype = dtype or torch.complex128
        self.gate_count = 0
        self.total_simulation_time_ns = 0

        self.amplitudes = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        self.amplitudes[0] = torch.tensor(1.0 + 0.0j, dtype=self.dtype, device=self.device)

    def reset(self):
        self.amplitudes.zero_()
        self.amplitudes[0] = torch.tensor(1.0 + 0.0j, dtype=self.dtype, device=self.device)
        self.gate_count = 0

    def get_all_probabilities(self) -> np.ndarray:
        """
        Probabilità |ψ_i|² di tutti gli stati base, calcolate a blocchi.

        Calcolare `torch.abs(state)**2` in un colpo solo creerebbe temporanei
        grandi quanto lo stato (più la normalizzazione), raddoppiando il picco
        di VRAM e causando OOM per stati grandi (es. 28 qubit). Qui si procede
        a blocchi: ogni segmento viene ridotto e trasferito su host, mantenendo
        il temporaneo GPU entro _CHUNK_TARGET_BYTES.
        """
        n = self.amplitudes.numel()
        out = np.empty(n, dtype=np.float64)
        chunk = max(1, _CHUNK_TARGET_BYTES // 16)
        total = 0.0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            seg = self.amplitudes[s:e]
            # |z|² = Re² + Im²: evita di materializzare un complesso intermedio.
            p = seg.real * seg.real + seg.imag * seg.imag
            total += float(p.sum().item())
            out[s:e] = p.detach().cpu().numpy()
        if total > 1e-15:
            out /= total
        return out

    def measure_all(self, collapse: bool = True) -> int:
        probs = self.get_all_probabilities()
        outcome = int(np.random.choice(len(probs), p=probs))
        if collapse:
            self.amplitudes.zero_()
            self.amplitudes[outcome] = torch.tensor(1.0 + 0.0j, dtype=self.dtype, device=self.device)
        return outcome

    def sample_all(self, shots: int) -> List[int]:
        probs = self.get_all_probabilities()
        return np.random.choice(len(probs), size=shots, p=probs).tolist()

    def fidelity(self, other_statevector) -> float:
        if isinstance(other_statevector, np.ndarray):
            other = torch.tensor(other_statevector, dtype=self.dtype, device=self.device)
        else:
            other = other_statevector.to(self.device)
        overlap = torch.abs(torch.vdot(self.amplitudes, other)) ** 2
        return float(overlap.detach().cpu().item())

    def to_numpy(self) -> np.ndarray:
        return self.amplitudes.detach().cpu().numpy()

    def get_bloch_vector(self, target_qubit: int) -> tuple[float, float, float]:
        """Calcola il vettore di Bloch per un qubit specifico via GPU."""
        n = self.n_qubits
        bit_pos = n - 1 - target_qubit
        mask = 1 << bit_pos

        indices = torch.arange(self.state_dim, device=self.device)
        probs = torch.abs(self.amplitudes) ** 2

        # Z = P(bit=0) - P(bit=1)
        signs = torch.where((indices & mask) == 0, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device))
        z_exp = float(torch.sum(signs * probs).item())

        # X, Y: overlap tra |bit=0⟩ e |bit=1⟩ per il qubit target
        i_idx = torch.where((indices & mask) == 0)[0]
        j_idx = i_idx | mask
        overlap = torch.sum(torch.conj(self.amplitudes[i_idx]) * self.amplitudes[j_idx])
        x_exp = float(2.0 * overlap.real.item())
        y_exp = float(2.0 * overlap.imag.item())

        return (x_exp, y_exp, z_exp)

    @staticmethod
    def _n_chunks(numel: int) -> int:
        """Numero di blocchi per mantenere il temporaneo entro _CHUNK_TARGET_BYTES."""
        total_bytes = numel * 16  # complex128
        return max(1, math.ceil(total_bytes / _CHUNK_TARGET_BYTES))

    def _apply_einsum_chunked(self, A, equation: str, M, spectator_axes):
        """
        Applica `einsum(equation, M, A)` in-place su A, a blocchi lungo l'asse
        spettatore più grande, per limitare il picco di memoria.

        A viene modificato in-place (è una view dello statevector). Ogni blocco
        legge una fetta, calcola il nuovo valore e lo riscrive nella stessa
        fetta: l'einsum materializza l'output prima dell'assegnazione, quindi
        non c'è aliasing. Il picco aggiuntivo è ~una fetta invece dell'intero
        stato.
        """
        # Scegli come asse da spezzare quello spettatore (non-gate) più grande.
        chunk_axis = max(spectator_axes, key=lambda ax: A.shape[ax])
        axis_len = A.shape[chunk_axis]
        n_chunks = min(self._n_chunks(A.numel()), axis_len)

        if n_chunks <= 1:
            # Stato piccolo: un solo passaggio (picco 2×, accettabile).
            A[...] = torch.einsum(equation, M, A)
            return

        for k in range(n_chunks):
            s = (k * axis_len) // n_chunks
            e = ((k + 1) * axis_len) // n_chunks
            if s == e:
                continue
            idx = [slice(None)] * A.ndim
            idx[chunk_axis] = slice(s, e)
            idx = tuple(idx)
            A[idx] = torch.einsum(equation, M, A[idx])

    def apply_gate(self, matrix: np.ndarray, target_qubit: int):
        """Applica un gate single-qubit 2x2 (einsum a blocchi, basso picco VRAM)."""
        if matrix.shape != (2, 2):
            raise ValueError("Single-qubit gate must be 2x2")

        M = torch.tensor(matrix, dtype=self.dtype, device=self.device)

        left_dim = 1 << target_qubit
        right_dim = 1 << (self.n_qubits - 1 - target_qubit)

        A = self.amplitudes.view(left_dim, 2, right_dim)
        # Assi spettatori: 0 (left) e 2 (right). L'asse 1 è il qubit target.
        self._apply_einsum_chunked(A, 'ab,lbr->lar', M, spectator_axes=(0, 2))
        self.gate_count += 1

    def apply_two_qubit_gate(self, matrix: np.ndarray, qubit_a: int, qubit_b: int):
        """Applica un gate 4x4 su due qubit (einsum a blocchi, basso picco VRAM)."""
        if matrix.shape != (4, 4):
            raise ValueError("Two-qubit gate must be 4x4")
        if qubit_a == qubit_b:
            raise ValueError("Two-qubit gate requires distinct qubits")

        M = torch.tensor(matrix, dtype=self.dtype, device=self.device).view(2, 2, 2, 2)

        q_min = min(qubit_a, qubit_b)
        q_max = max(qubit_a, qubit_b)

        d1 = 1 << q_min
        d2 = 2
        d3 = 1 << (q_max - q_min - 1)
        d4 = 2
        d5 = 1 << (self.n_qubits - 1 - q_max)

        A = self.amplitudes.view(d1, d2, d3, d4, d5)

        # Assi spettatori: 0 (d1), 2 (d3), 4 (d5). Gli assi 1 e 3 sono i qubit.
        if qubit_a < qubit_b:
            self._apply_einsum_chunked(A, 'abcd,lcmdo->lambo', M, spectator_axes=(0, 2, 4))
        else:
            self._apply_einsum_chunked(A, 'abcd,ldmco->lbmao', M, spectator_axes=(0, 2, 4))

        self.gate_count += 1
