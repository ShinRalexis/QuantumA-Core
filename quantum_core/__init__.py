"""
QuantumA Core — Simulatore di circuiti quantistici con rumore realistico.

Architettura:
- Statevector Engine: stato puro vettorizzato con NumPy (einsum), backend GPU
  opzionale via PyTorch/CUDA
- Density Matrix Engine: stati misti e canali di rumore
- Kraus Operators: modelli di decoerenza T1/T2 calibrati su profili hardware
- Circuit Builder: costruzione, ottimizzazione di base e serializzazione
"""

__version__ = "1.0.0"
__author__ = "QuantumA Core Team"
