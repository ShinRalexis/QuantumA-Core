
import requests
import json

def run_bv_test():
    url = "http://localhost:8227/simulate"
    
    # Segreto: 11010 (5 bit)
    secret = "11010"
    n_bits = len(secret)
    n_qubits = n_bits + 1 # 6 Qubits totali
    
    circuit = []
    
    # Step 1: H su tutti i qubit di input
    for i in range(n_bits):
        circuit.append({"gate": "H", "qubits": [i]})
    
    # Step 2: Qubit ausiliario in stato |-> (X poi H)
    circuit.append({"gate": "X", "qubits": [n_bits]})
    circuit.append({"gate": "H", "qubits": [n_bits]})
    
    # Step 3: Oracolo (Kickback della fase)
    # Applichiamo CNOT tra il qubit i e l'ausiliario se secret[i] == '1'
    for i, bit in enumerate(secret):
        if bit == '1':
            circuit.append({"gate": "CNOT", "qubits": [i, n_bits]})
            
    # Step 4: H finale per collassare la fase in ampiezza
    for i in range(n_bits):
        circuit.append({"gate": "H", "qubits": [i]})
        
    payload = {
        "n_qubits": n_qubits,
        "mode": "statevector",
        "backend": "cuda",
        "instructions": circuit,
        "shots": 1024
    }
    
    print(f"--- Test Accademico: Bernstein-Vazirani (6 Qubits) ---")
    print(f"Obiettivo: Estrarre il segreto '{secret}' con una sola query.")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            # Lo stato finale sara '110101' (il sesto bit e l'ausiliario)
            full_state = result['top_states'][0]['state']
            found_secret = full_state[:n_bits]
            prob = result['top_states'][0]['probability']
            
            print(f"\nRisultato:")
            print(f"Segreto estratto: {found_secret}")
            print(f"Probabilita: {prob:.2%}")
            print(f"Tempo GPU: {result['simulation_time_ms']:.2f}ms")
            
            if found_secret == secret and prob > 0.99:
                print("\nVERIFICA SUPERATA: Segreto identificato correttamente.")
            else:
                print(f"\nERRORE: Risultato inaspettato ({found_secret})")
        else:
            print(f"Errore Server: {response.text}")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    run_bv_test()
