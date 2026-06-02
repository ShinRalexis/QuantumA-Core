
import torch
import time
import numpy as np
from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

def run_pro_validation():
    # 26 qubit = 2^26 = 67M ampiezze complex128 = ~1.07 GB di stato
    n_qubits = 26
    print(f"--- AVVIO SIMULAZIONE PROFESSIONALE: {n_qubits} QUBIT ---")
    print(f"Stati in memoria: {1 << n_qubits:,}")
    
    if not torch.cuda.is_available():
        print("ERRORE: CUDA non disponibile!")
        return
        
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    
    # Inizializziamo il simulatore in modalità CUDA
    sim = QuantumSimulator(n_qubits, mode=SimulationMode.STATEVECTOR, backend='cuda')
    
    # Creiamo un circuito denso
    qc = QuantumCircuit(n_qubits)
    layers = 10
    
    # Costruzione circuito
    for _ in range(layers):
        for i in range(n_qubits):
            qc.h(i)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i+1)
            
    print(f"Circuito generato: {qc.total_gates} gate totali.")
    print("\n>>> ESECUZIONE SU GPU (Monitora Task Manager ora)...")
    
    start = time.time()
    sim.run_circuit(qc)
    end = time.time()
    
    duration = end - start
    print(f"\nCOMPLETATO IN: {duration:.4f} secondi")
    print(f"Velocità: {qc.total_gates / duration:.2f} gate/sec")
    print(f"Stato finale device: {sim.engine.amplitudes.device}")
    
    # Verifica che lo stato sia normalizzato (prova che i calcoli sono corretti)
    norm = torch.sum(torch.abs(sim.engine.amplitudes)**2).item()
    print(f"Norma dello stato: {norm:.4f} (Atteso: 1.0000)")

if __name__ == "__main__":
    run_pro_validation()
