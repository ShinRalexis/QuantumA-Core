
import sys
import time
from pathlib import Path

# Aggiungi il root del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

def gpu_stress_test(n_qubits=28):
    print(f"=== QuantumA Core: GPU Stress Test ({n_qubits} Qubits) ===")
    
    try:
        # Inizializzazione forzata su CUDA
        sim = QuantumSimulator(n_qubits=n_qubits, mode=SimulationMode.STATEVECTOR, backend='cuda')
        
        if sim.backend != 'cuda':
            print("ERRORE: Il simulatore non ha attivato il backend CUDA.")
            print("Verifica che PyTorch e i driver NVIDIA siano installati correttamente.")
            return

        print(f"Backend attivato: {sim.backend.upper()}")
        print(f"Dimensione stato: 2^{n_qubits} ({1 << n_qubits} ampiezze)")
        print(f"Memoria stimata: ~{(1 << n_qubits) * 16 / (1024**2):.2f} MB")
        
        # Creazione di un circuito con gate pesanti
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
        
        print("\nAvvio simulazione sulla GPU...")
        start = time.perf_counter()
        result = sim.run_circuit(qc, shots=1)
        end = time.perf_counter()
        
        print(f"Simulazione completata in: {(end - start):.4f} secondi")
        print(f"Tempo core simulatore: {result.simulation_time_ms:.2f} ms")
        print("\nControlla ora il Task Manager / NVIDIA-SMI: dovresti aver visto un picco di calcolo e memoria.")

    except Exception as e:
        print(f"Errore durante lo stress test: {e}")

if __name__ == "__main__":
    gpu_stress_test(28)
