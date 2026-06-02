
import requests
import json
import time

def run_grover_perfect():
    url = "http://localhost:8227/simulate"
    
    # Algoritmo di Grover PERFETTO per 2 Qubit (Trova |11>)
    # Matematicamente, con 1 iterazione, la probabilita deve essere 100%
    circuit = [
        # 1. Superposizione
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        
        # 2. Oracolo per |11> (Inverte la fase di |11>)
        {"gate": "CZ", "qubits": [0, 1]},
        
        # 3. Diffusore (Inversione rispetto alla media)
        {"gate": "H", "qubits": [0]}, {"gate": "H", "qubits": [1]},
        {"gate": "X", "qubits": [0]}, {"gate": "X", "qubits": [1]},
        {"gate": "CZ", "qubits": [0, 1]},
        {"gate": "X", "qubits": [0]}, {"gate": "X", "qubits": [1]},
        {"gate": "H", "qubits": [0]}, {"gate": "H", "qubits": [1]}
    ]
    
    payload = {
        "n_qubits": 2,
        "mode": "statevector",
        "backend": "cuda",
        "instructions": circuit,
        "shots": 1024
    }
    
    print("--- Test Accademico: Grover Perfetto (2 Qubit) ---")
    print("Obiettivo: Identificare lo stato |11> con probabilita 100%.")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            top_state = result['top_states'][0]
            print(f"\nRisultato Simulator:")
            print(f"Stato dominante: |{top_state['state']}>")
            print(f"Probabilita: {top_state['probability']:.2%}")
            print(f"Tempo GPU: {result['simulation_time_ms']:.2f}ms")
            
            if top_state['state'] == "11" and top_state['probability'] > 0.99:
                print("\n✅ VERIFICA ACCADEMICA SUPERATA: Precisione del 100% confermata.")
            else:
                print("\n❌ ERRORE: La probabilita non e quella attesa per un sistema puro.")
        else:
            print(f"Errore Server: {response.text}")
            
    except Exception as e:
        print(f"Errore connessione: {e}")

if __name__ == "__main__":
    run_grover_perfect()
