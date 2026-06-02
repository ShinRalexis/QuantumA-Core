from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit
import time
import math


def build_circuit(n, layers):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for layer in range(layers):
        angle = (layer + 1) * math.pi / 17.0
        for q in range(n):
            if (q + layer) % 3 == 0:
                qc.ry(q, angle)
            elif (q + layer) % 3 == 1:
                qc.rz(q, angle / 2)
            else:
                qc.rx(q, angle / 3)
        if layer % 2 == 0:
            for q in range(0, n - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in range(1, n - 1, 2):
                qc.cx(q, q + 1)
        for q in range(0, n // 2):
            target = n - 1 - q
            if q < target:
                qc.cz(q, target)
    return qc


for n in (20, 24, 28):
    layers = 10 if n == 20 else 14 if n == 24 else 18
    qc = build_circuit(n, layers)
    sim = QuantumSimulator(n_qubits=n, mode=SimulationMode.STATEVECTOR, hardware_profile='superconducting', backend='cuda')
    t0 = time.perf_counter()
    res = sim.run_circuit(qc, shots=256)
    elapsed = time.perf_counter() - t0
    print(f'n={n} layers={layers} gates={qc.total_gates} elapsed_s={elapsed:.4f} backend={sim.backend} device={sim.engine.amplitudes.device} norm={float((sim.engine.amplitudes.abs()**2).sum().item()):.6f}')
