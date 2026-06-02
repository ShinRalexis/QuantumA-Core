# QuantumA Core - Guida all'Integrazione per Agenti AI / LLM

Questa guida è un "vocabolario" tecnico per agenti autonomi e LLM, pensato per garantire una comunicazione senza errori con l'API REST di QuantumA Core. Leggi attentamente prima di pianificare l'esecuzione di algoritmi quantistici.

---

## 1. Regole Generali di Comunicazione
- **Indirizzo API Base:** L'API risiede tipicamente su `http://127.0.0.1:8227` (o `http://host.docker.internal:8227` se l'agente gira dentro Docker).
- **Formato Richiesta:** JSON rigoroso (`application/json`).
- **Interpretazione Risultati:** Il campo `top_states` contiene gli stati con le probabilità maggiori. Gli stati sono restituiti come **stringhe binarie** (es. `"0101"`). Non convertire le stringhe binarie in interi prima di aver controllato il qubit ordering.

---

## 2. L'Endpoint `/simulate` e la Sintassi dei Gate
L'errore più comune commesso dagli agenti è inviare array di istruzioni malformati.
Il payload corretto per `/simulate` richiede:
1. `n_qubits` (intero)
2. `mode` (stringa: `"statevector"`, `"density_matrix"`, `"monte_carlo"`)
3. `instructions` (Array di Oggetti)

### ⚠️ Regola d'Oro per l'array `instructions`:
Ogni gate **DEVE** essere un oggetto dizionario con le chiavi `gate` e `qubits`. **MAI** passare un array di array, e **MAI** inserire angoli di rotazione nell'array dei qubit.

**Corretto (Rotazione Y di π/4 sul qubit 2):**
```json
{
  "gate": "RY", 
  "qubits": [2], 
  "params": {"theta": 0.785398}
}
```

**Sbagliato (Allucinazione LLM comune):**
```json
// ERRORE 1: Inserire il parametro nei qubit
{"gate": "RY", "qubits": [2, 0.785398]} 

// ERRORE 2: Formato lista posizionale invece di dizionario
["RY", [2], {"theta": 0.785398}] 
```

---

## 3. Risoluzione di Algoritmi Specifici

### A. Algoritmo di Grover
**Non provare MAI a decomporre un oracolo (Multi-Controlled-Z) a mano usando `/simulate`.** Costruire un CCZ generalizzato per $N$ qubit usando solo CNOT e H è computazionalmente inefficace e porta a bug logici.
**Soluzione:** Usa **SEMPRE** l'endpoint dedicato `/grover`.
QuantumA Core possiede un risolutore matematico puro integrato che lavora in $O(1)$ gate sul vettore di stato.

**Payload Ottimale:**
```json
POST /grover
{
  "n_qubits": 4,
  "target_state": 5,
  "iterations": 3,
  "mode": "statevector"
}
```
*Nota Agente: Per calcolare le `iterations` ottimali, usa SEMPRE la formula matematica $ \frac{\pi}{4} \sqrt{2^{n\_qubits}} $.*

### B. Misurazioni su Assi Arbitrari (es. Test CHSH)
Per misurare in una base diversa da Z (es. X o una base ruotata), l'agente deve applicare la **rotazione inversa** prima di effettuare la misurazione finale implicita.
Se vuoi misurare un qubit lungo l'asse ruotato di un angolo $\theta$, devi applicare la rotazione inversa $-\theta$.

**Esempio per il test CHSH (Disuguaglianza di Bell):**
Per dimostrare la violazione massimale ($S = 2\sqrt{2}$), gli angoli di osservazione teorici ottimali per lo stato $|\Phi^+\rangle$ sono:
- Alice: $a = 0$ (Base Z), $a' = \pi/2$ (Base X)
- Bob: $b = \pi/4$, $b' = 3\pi/4$

Poiché il simulatore deve ruotare l'osservabile indietro sulla base computazionale Z, nel circuito l'agente **DEVE NEGARE** questi angoli nel parametro `theta` dei gate `RY`:
```json
// Misurazione di Alice in base X (a' = π/2)
{"gate": "RY", "qubits": [0], "params": {"theta": -1.57079}}

// Misurazione di Bob (b' = 3π/4)
{"gate": "RY", "qubits": [1], "params": {"theta": -2.35619}}
```
*Errore tipico dell'LLM: Usare $-\pi/4$ invece di $3\pi/4$ per Bob, ottenendo $S=0$.*

---

## 4. Glossario dei Gate Supportati
Usa esattamente queste stringhe per il campo `gate` (l'API è comunque case-insensitive):
- **Singolo Qubit:** `"H"`, `"X"`, `"Y"`, `"Z"`, `"S"`, `"T"`
- **Rotazioni Singolo Qubit (richiedono `theta` per RX/RY, `phi` per RZ):** `"RX"`, `"RY"`, `"RZ"`
- **Due Qubit:** `"CNOT"` (o `"CX"`), `"CZ"`, `"SWAP"`, `"ISWAP"`, `"CRZ"` (richiede `phi`), `"CPHASE"` (richiede `theta`), `"RZZ"` (richiede `theta`).
- **Tre Qubit:** `"CCNOT"` (o `"TOFFOLI"`).

*Nota: il backend GPU applica gate a 1–2 qubit; i gate a 3 qubit (CCNOT) girano su CPU. Per un Multi-Controlled-Z su molti qubit (oltre il Toffoli) non esiste un gate dedicato: usa l'endpoint `/grover`, che applica l'oracolo direttamente sullo statevector.*

## 5. Lettura dello Stato del Sistema
Prima di operazioni massive, esegui una GET su `/status/gpu`.
- Se `cuda_available` è `true`, il sistema sfrutta tensori PyTorch nella VRAM.
- In modalità `statevector` il limite su una GPU da 8 GB è **28 qubit** (oltre si passa automaticamente a CPU). In modalità `density_matrix` (che calcola rumore e operatori di Kraus) il limite di memoria stringe a ~16-18 qubit. Adatta i tuoi test di conseguenza.
