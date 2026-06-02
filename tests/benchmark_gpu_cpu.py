import time
from statistics import mean

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit


def build_bell_chain(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def run_benchmark(n_qubits_list=(8, 10, 12, 14), shots=1024):
    print("QuantumA Core Benchmark")
    print("=" * 40)

    for backend in ("cpu", "auto", "cuda"):
        print(f"\nBackend: {backend}")
        for n in n_qubits_list:
            qc = build_bell_chain(n)
            times = []
            used_backend = None
            for _ in range(5):
                sim = QuantumSimulator(
                    n_qubits=n,
                    mode=SimulationMode.STATEVECTOR,
                    hardware_profile="superconducting",
                    backend=backend,
                )
                t0 = time.perf_counter()
                result = sim.run_circuit(qc, shots=shots)
                dt = (time.perf_counter() - t0) * 1000
                times.append(dt)
                used_backend = getattr(result, "noise_summary", {}).get("backend", sim.backend)
            print(
                f"  n={n:2d} | avg_ms={mean(times):8.3f} | backend_used={used_backend}"
            )


if __name__ == "__main__":
    run_benchmark()
