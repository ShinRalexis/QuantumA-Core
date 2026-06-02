
import sys
import time
import numpy as np
from pathlib import Path

# Aggiungi il root del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

def run_qaoa_maxcut_28q():
    n_qubits = 28
    print(f"=== QuantumA Research: QAOA Max-Cut Benchmark ({n_qubits} Qubits) ===")
    
    # 1. Generazione Grafo (Erdos-Renyi casuale)
    # Usiamo un seed fisso per riproducibilità scientifica
    np.random.seed(42)
    edges = []
    density = 0.3
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if np.random.rand() < density:
                edges.append((i, j))
    
    print(f"Grafo generato: {n_qubits} nodi, {len(edges)} archi.")

    # 2. Parametri QAOA (P=1)
    gamma = 0.3927  # Angolo di fase (Cost)
    beta = 0.1963   # Angolo di mixing
    
    # 3. Costruzione Circuito
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Sovrapposizione uniforme
    for i in range(n_qubits):
        qc.h(i)
    
    # Step 2: Cost Hamiltonian (ZZ interactions per ogni arco)
    # H_C = sum_{i,j in E} 0.5 * (I - ZiZj)
    for u, v in edges:
        # Implementazione ZZ(gamma)
        qc.cx(u, v)
        qc.rz(v, phi=gamma)
        qc.cx(u, v)
    
    # Step 3: Mixing Hamiltonian (RX rotations)
    # H_B = sum_{i} Xi
    for i in range(n_qubits):
        qc.rx(i, theta=2*beta)
        
    print(f"Circuito QAOA costruito: {qc.total_gates} gate totali.")
    
    # 4. Simulazione su GPU
    sim = QuantumSimulator(n_qubits=n_qubits, mode=SimulationMode.STATEVECTOR, backend='cuda')
    
    print("\n[GPU] Esecuzione simulazione statevector (4GB VRAM)...")
    start_time = time.perf_counter()
    result = sim.run_circuit(qc, shots=16384)
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    
    # 5. Analisi Risultati
    top_states = result.most_likely_states(5)
    best_bitstring, best_prob = top_states[0]
    
    # Calcolo del valore del cut per la bitstring migliore
    def calculate_cut(bitstr, edges):
        cut = 0
        for u, v in edges:
            if bitstr[u] != bitstr[v]:
                cut += 1
        return cut

    best_cut_val = calculate_cut(best_bitstring, edges)
    
    print(f"\n--- RISULTATI SCIENTIFICI ---")
    print(f"Tempo di esecuzione GPU: {result.simulation_time_ms:.2f} ms")
    print(f"Tempo totale (incluso overhead): {exec_time:.2f} s")
    print(f"Soluzione Ottimale Approssimata: {best_bitstring}")
    print(f"Valore del Cut (Max-Cut): {best_cut_val} / {len(edges)} archi")
    print(f"Probabilità di misura: {best_prob*100:.4f}%")
    print(f"Fedeltà del circuito stimata: {result.circuit_fidelity:.6f}")
    
    # Salvataggio dati per l'abstract
    with open("QAOA_RESULTS_28Q.txt", "w") as f:
        f.write(f"QAOA MAX-CUT 28-QUBIT REPORT\n")
        f.write(f"Nodes: {n_qubits}\n")
        f.write(f"Edges: {len(edges)}\n")
        f.write(f"GPU Core Time: {result.simulation_time_ms:.2f} ms\n")
        f.write(f"Best Bitstring: {best_bitstring}\n")
        f.write(f"Max Cut Value: {best_cut_val}\n")
        f.write(f"Confidence: {best_prob:.6f}\n")
        f.write(f"Circuit Fidelity: {result.circuit_fidelity}\n")

if __name__ == "__main__":
    run_qaoa_maxcut_28q()
