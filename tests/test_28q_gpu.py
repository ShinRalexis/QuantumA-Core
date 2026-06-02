from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit
import time

n = 28
qc = QuantumCircuit(n)
qc.h(0)
qc.cx(0, 1)

sim = QuantumSimulator(n_qubits=n, mode=SimulationMode.STATEVECTOR, hardware_profile='superconducting', backend='cuda')
start = time.perf_counter()
res = sim.run_circuit(qc, shots=1)
elapsed = time.perf_counter() - start

print('backend=', sim.backend)
print('device=', sim.engine.amplitudes.device)
print('elapsed_s=', round(elapsed, 4))
print('top_states=', res.most_likely_states(5))
print('norm=', round(float((sim.engine.amplitudes.abs()**2).sum().item()), 6))
