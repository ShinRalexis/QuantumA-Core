"""
Benchmark — Performance testing del simulatore quantistico.

Testa le performance su diversi numeri di qubit e tipi di circuito:
- Scaling dello statevector (fino a ~28 qubit)
- Scaling della matrice densità (fino a ~18 qubit)
- Algoritmi standard (Grover, QFT, Random Circuits)
- Confronto modalità di simulazione e backend CPU/GPU
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit


@dataclass
class BenchmarkResult:
    """Risultato di un benchmark."""
    n_qubits: int
    circuit_type: str
    mode: str
    gate_count: int
    two_qubit_gates: int
    simulation_time_ms: float
    shots_per_second: float
    circuit_fidelity: float
    estimated_error_rate: float
    peak_memory_mb: float = 0.0
    
    def throughput(self) -> float:
        """Gate throughput (gates/sec)."""
        if self.simulation_time_ms == 0:
            return float('inf')
        return self.gate_count / (self.simulation_time_ms / 1000)


def create_random_circuit(n_qubits: int, n_gates: int, 
                         two_qubit_ratio: float = 0.3) -> QuantumCircuit:
    """Crea un circuito casuale per benchmarking."""
    circuit = QuantumCircuit(n_qubits)
    
    single_gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
    two_qubit_gates = ['CNOT', 'CZ', 'SWAP']
    
    for _ in range(n_gates):
        if np.random.random() < two_qubit_ratio and n_qubits >= 2:
            q1 = np.random.randint(0, n_qubits)
            q2 = np.random.randint(0, n_qubits)
            while q2 == q1:
                q2 = np.random.randint(0, n_qubits)
            
            gate = np.random.choice(two_qubit_gates)
            method = {'cnot': 'cx'}.get(gate.lower(), gate.lower())
            getattr(circuit, method)(q1, q2)
        else:
            q = np.random.randint(0, n_qubits)
            gate = np.random.choice(single_gates)
            getattr(circuit, gate.lower())(q)
    
    return circuit


def run_scaling_benchmark(max_qubits: int = 25, 
                          shots: int = 1024) -> List[BenchmarkResult]:
    """Benchmark scaling statevector vs qubit count."""
    results = []
    
    print("=" * 70)
    print("📊 SCALING BENCHMARK — Statevector Simulation")
    print("=" * 70)
    
    for n_qubits in range(2, min(max_qubits + 1, 26)):
        # Calcola memoria stimata: 2^n × 16 bytes (complex128)
        memory_bytes = (2 ** n_qubits) * 16
        memory_mb = memory_bytes / (1024 * 1024)
        
        print(f"\n{n_qubits} qubit | Memory: {memory_mb:.1f} MB", end="")
        
        # Circuit con depth proporzionale a n_qubits
        circuit = create_random_circuit(
            n_qubits, 
            n_gates=n_qubits * 10,
            two_qubit_ratio=0.3
        )
        
        try:
            simulator = QuantumSimulator(n_qubits, SimulationMode.STATEVECTOR)
            
            start = time.perf_counter()
            result = simulator.run_circuit(circuit, shots=shots)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            res = BenchmarkResult(
                n_qubits=n_qubits,
                circuit_type="random",
                mode="statevector",
                gate_count=circuit.total_gates,
                two_qubit_gates=circuit.two_qubit_gates,
                simulation_time_ms=round(elapsed_ms, 3),
                shots_per_second=result.shots_completed / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
                circuit_fidelity=result.circuit_fidelity,
                estimated_error_rate=result.estimated_error_rate,
                peak_memory_mb=memory_mb,
            )
            
            results.append(res)
            
            print(f" | ⏱ {elapsed_ms:.1f}ms | 🚀 {res.throughput():,.0f} gates/sec")
        
        except MemoryError:
            print(f" | ❌ OUT OF MEMORY (need ~{memory_mb:.0f} MB)")
            break
    
    return results


def run_algorithm_benchmarks(shots: int = 4096):
    """Benchmark algoritmi quantistici standard."""
    results = []
    
    algorithms = [
        ("Bell State", 2, "bell"),
        ("GHZ (4 qubit)", 4, "ghz"),
        ("Grover (3 qubit)", 3, "grover"),
        ("Random Circuit (10 qubit)", 10, "random"),
    ]
    
    print("\n" + "=" * 70)
    print("⚡ ALGORITHM BENCHMARK — Various Quantum Algorithms")
    print("=" * 70)
    
    for name, n_qubits, algo_type in algorithms:
        print(f"\n📐 {name}")
        
        # Test statevector
        try:
            sim = QuantumSimulator(n_qubits, SimulationMode.STATEVECTOR, "superconducting")
            
            start = time.perf_counter()
            if algo_type == "bell":
                result = sim.run_bell_state(shots=shots)
            elif algo_type == "ghz":
                result = sim.run_ghz_state(shots=shots)
            elif algo_type == "grover":
                result = sim.run_grover(n_iterations=1, shots=shots)
            else:  # random
                circuit = create_random_circuit(n_qubits, n_qubits * 20)
                result = sim.run_circuit(circuit, shots=shots)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            print(f"   Statevector: {elapsed_ms:.2f}ms | "
                  f"Fidelity: {result.circuit_fidelity:.4f} | "
                  f"Error: {result.estimated_error_rate:.4%}")
            
            # Print top results
            for state, prob in result.most_likely_states(3):
                print(f"     → {state}: {prob:.4f}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")


def run_noise_comparison(n_qubits: int = 5, shots: int = 1024):
    """Confronta le tre modalità di simulazione con rumore."""
    
    circuit = create_random_circuit(n_qubits, n_qubits * 8)
    
    modes = [
        (SimulationMode.STATEVECTOR, "Statevector (no noise)"),
        (SimulationMode.DENSITY_MATRIX, "Density Matrix (with noise)"),
        (SimulationMode.MONTE_CARLO, "Monte Carlo Wave Function"),
    ]
    
    print("\n" + "=" * 70)
    print("🔬 NOISE COMPARISON — n_qubits=" + str(n_qubits))
    print("=" * 70)
    
    for mode, label in modes:
        try:
            sim = QuantumSimulator(n_qubits, mode, "superconducting")
            
            start = time.perf_counter()
            result = sim.run_circuit(circuit, shots=shots)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            print(f"\n📌 {label}")
            print(f"   ⏱ Time: {elapsed_ms:.2f}ms")
            print(f"   🎯 Fidelity: {result.circuit_fidelity:.6f}")
            print(f"   ⚠ Error rate: {result.estimated_error_rate:.4%}")
            
            if result.most_likely_states(3):
                top = result.most_likely_states(3)
                for state, prob in top:
                    print(f"     → {state}: {prob:.6f}")
        
        except Exception as e:
            print(f"\n📌 {label} ❌ Error: {e}")


def run_platform_comparison(n_qubits: int = 4, shots: int = 1024):
    """Confronta diversi profili hardware."""
    
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.h(i)
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    
    platforms = [
        "superconducting",
        "trapped_ion", 
        "silicon_spin",
        "neutral_atom",
    ]
    
    print("\n" + "=" * 70)
    print("🖥️ PLATFORM COMPARISON — GHZ State (4 qubit)")
    print("=" * 70)
    
    for platform in platforms:
        sim = QuantumSimulator(n_qubits, SimulationMode.STATEVECTOR, platform)
        
        start = time.perf_counter()
        result = sim.run_circuit(circuit, shots=shots)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"\n📌 {platform.replace('_', ' ').title()}")
        stats = sim.get_stats()
        
        # Print hardware params
        for q_idx in range(n_qubits):
            params = sim.qubit_params[q_idx]
            print(f"   Q{q_idx}: T1={params.t1:.0f}μs, "
                  f"T2={params.t2:.0f}μs, "
                  f"F_single={params.fidelity_single:.6f}, "
                  f"F_two={params.fidelity_two:.6f}")
        
        print(f"   ⏱ Time: {elapsed_ms:.2f}ms")
        print(f"   🎯 Fidelity: {result.circuit_fidelity:.6f}")


def main():
    """Esegue tutti i benchmark."""
    
    print("""
╔═══════════════════════════════════════════════════╗
║   QuantumA Core — Benchmark Suite               ║
║   Simulatore di Chip Quantistico Realistico       ║
╚═══════════════════════════════════════════════════╝
    """)
    
    # 1. Scaling benchmark (statevector)
    scaling_results = run_scaling_benchmark(max_qubits=20, shots=512)
    
    # 2. Algorithm benchmarks
    run_algorithm_benchmarks(shots=4096)
    
    # 3. Noise comparison
    run_noise_comparison(n_qubits=5, shots=1024)
    
    # 4. Platform comparison
    run_platform_comparison(n_qubits=4, shots=1024)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    if scaling_results:
        best = max(scaling_results, key=lambda r: r.throughput())
        print(f"\n🏆 Best throughput: {best.throughput():,.0f} gates/sec "
              f"({best.n_qubits} qubits)")
        
        # Memory usage table
        print(f"\n{'Qubits':<10} {'Memory (MB)':<15} {'Time (ms)':<15} {'Gates/sec'}")
        print("-" * 55)
        for r in scaling_results:
            print(f"{r.n_qubits:<10} {r.peak_memory_mb:<15.1f} "
                  f"{r.simulation_time_ms:<15.3f} "
                  f"{r.throughput():,.0f}")


if __name__ == "__main__":
    main()
