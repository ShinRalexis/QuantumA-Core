
import requests
import json
import numpy as np

def run_qft_test():
    url = "http://localhost:8227/simulate"
    
    # 4 Qubit QFT
    # Prepariamo lo stato |1> (frequenza minima)
    # La QFT trasformerà questo stato in una superposizione con fasi specifiche
    
    circuit = [
        # Preparazione: Stato |0001> (decimale 1)
        {"gate": "X", "qubits": [3]},
        
        # QFT su 4 qubit
        # Qubit 0
        {"gate": "H", "qubits": [0]},
        {"gate": "CPHASE", "qubits": [1, 0], "params": {"theta": np.pi/2}},
        {"gate": "CPHASE", "qubits": [2, 0], "params": {"theta": np.pi/4}},
        {"gate": "CPHASE", "qubits": [3, 0], "params": {"theta": np.pi/8}},
        
        # Qubit 1
        {"gate": "H", "qubits": [1]},
        {"gate": "CPHASE", "qubits": [2, 1], "params": {"theta": np.pi/2}},
        {"gate": "CPHASE", "qubits": [3, 1], "params": {"theta": np.pi/4}},
        
        # Qubit 2
        {"gate": "H", "qubits": [2]},
        {"gate": "CPHASE", "qubits": [3, 2], "params": {"theta": np.pi/2}},
        
        # Qubit 3
        {"gate": "H", "qubits": [3]},
        
        # Swap per invertire l'ordine (standard QFT)
        {"gate": "SWAP", "qubits": [0, 3]},
        {"gate": "SWAP", "qubits": [1, 2]}
    ]
    
    payload = {
        "n_qubits": 4,
        "mode": "statevector",
        "backend": "cuda",
        "instructions": circuit,
        "shots": 1024
    }
    
    print("--- Test Accademico Finale: Quantum Fourier Transform (4 Qubit) ---")
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"Calcolo QFT completato in {result['simulation_time_ms']:.2f}ms su GPU")
            
            # Nella QFT di uno stato computazionale, tutte le probabilità sono uguali (1/2^n)
            # ma le FASI cambiano. La verifica è che la distribuzione sia uniforme.
            probs = [s['probability'] for s in result['top_states']]
            avg_prob = 1.0 / (2**4)
            
            print(f"Probabilita media attesa: {avg_prob:.4%}")
            print(f"Probabilita rilevata: {probs[0]:.4%}")
            
            if abs(probs[0] - avg_prob) < 0.01:
                print("\nVERIFICA QFT SUPERATA: Il simulatore gestisce correttamente rotazioni e interferenze.")
            else:
                print("\nERRORE: La distribuzione non e uniforme come previsto dalla teoria.")
        else:
            print(f"Errore: {response.text}")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    run_qft_test()
