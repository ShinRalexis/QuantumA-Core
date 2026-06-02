from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit
import time
import math

n = 28
layers = 18
qc = QuantumCircuit(n)

# Stato iniziale molto entangled e profondo
for q in range(n):
    qc.h(q)

for layer in range(layers):
    angle = (layer + 1) * math.pi / 17.0
    # Rotazioni su tutti i qubit
    for q in range(n):
        if (q + layer) % 3 == 0:
            qc.ry(q, angle)
        elif (q + layer) % 3 == 1:
            qc.rz(q, angle / 2)
        else:
            qc.rx(q, angle / 3)

    # Entanglement a catena con pattern alternato
    if layer % 2 == 0:
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
    else:
        for q in range(1, n - 1, 2):
            qc.cx(q, q + 1)

    # Cross-entanglement a lunga distanza
    for q in range(0, n // 2):
        target = n - 1 - q
        if q < target:
            qc.cz(q, target)

sim = QuantumSimulator(
    n_qubits=n,
    mode=SimulationMode.STATEVECTOR,
    hardware_profile='superconducting',
    backend='cuda',
)

start = time.perf_counter()
res = sim.run_circuit(qc, shots=1024)
elapsed = time.perf_counter() - start

print('backend=', sim.backend)
print('device=', sim.engine.amplitudes.device)
print('elapsed_s=', round(elapsed, 4))
print('gate_count=', qc.total_gates)
print('top_states=', res.most_likely_states(10))
print('norm=', round(float((sim.engine.amplitudes.abs()**2).sum().item()), 6))
print('shots_completed=', res.shots_completed)
