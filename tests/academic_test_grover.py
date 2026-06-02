
import requests
import json
import time

def run_grover_test():
    # URL del server Docker
    url = "http://localhost:8227/simulate"
    
    # Costruiamo l'algoritmo di Grover per trovare lo stato |101> (5 decimale)
    # n_qubits = 3
    
    circuit = [
        # Step 1: Superposizione
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "H", "qubits": [2]},
        
        # Step 2: Oracolo per |101>
        {"gate": "X", "qubits": [1]},
        {"gate": "CZ", "qubits": [0, 2]}, # Phase flip se Q0 e Q2 sono 1
        {"gate": "X", "qubits": [1]},
        
        # Step 3: Diffusione (Inversione rispetto alla media)
        {"gate": "H", "qubits": [0]}, {"gate": "H", "qubits": [1]}, {"gate": "H", "qubits": [2]},
        {"gate": "X", "qubits": [0]}, {"gate": "X", "qubits": [1]}, {"gate": "X", "qubits": [2]},
        {"gate": "CZ", "qubits": [0, 2]},
        {"gate": "X", "qubits": [0]}, {"gate": "X", "qubits": [1]}, {"gate": "X", "qubits": [2]},
        {"gate": "H", "qubits": [0]}, {"gate": "H", "qubits": [1]}, {"gate": "H", "qubits": [2]}
    ]
    
    payload = {
        "n_qubits": 3,
        "mode": "statevector",
        "backend": "cuda",
        "instructions": circuit,
        "shots": 1024
    }
    
    print("--- Test Accademico: Algoritmo di Grover ---")
    print("Obiettivo: Identificare lo stato |101> tra 8 possibilità.")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Latenza API: {(end_time - start_time)*1000:.2f}ms")
            print(f"Tempo di calcolo GPU: {result['simulation_time_ms']:.2f}ms")
            print("\nClassifica stati (Top 3):")
            for i, s in enumerate(result['top_states'][:3]):
                status = "[SOLUZIONE]" if s['state'] == "101" else ""
                print(f"{i+1}. |{s['state']}> con probabilita {s['probability']:.2%} {status}")
            
            # Verifica scientifica
            solution_prob = next(s['probability'] for s in result['top_states'] if s['state'] == "101")
            if solution_prob > 0.5:
                print("\nVERIFICA ACCADEMICA SUPERATA: Grover ha amplificato correttamente la soluzione.")
            else:
                print("\nVERIFICA FALLITA: Probabilita troppo bassa.")
        else:
            print(f"Errore Server: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Errore di connessione: {e}")

if __name__ == "__main__":
    run_grover_test()
