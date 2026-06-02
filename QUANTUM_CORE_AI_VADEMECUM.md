# 🧠 QuantumA Core - Vademecum per Agenti AI
> **Direttiva di Sistema:** Leggi e rispetta questo protocollo prima di generare codice o circuiti per QuantumA Core.

Questo documento serve a prevenire "spirali di riscrittura" e scelte architetturali errate comuni negli agenti LLM meno esperti.

---

## 1. Architettura di Riferimento
- **Base URL:** `http://127.0.0.1:8227` (o `http://host.docker.internal:8227` in Docker).
- **Core Engine:** Doppio backend. GPU (PyTorch/CUDA) quando disponibile, CPU (NumPy/einsum) come fallback. Entrambi usano la stessa libreria di gate e producono risultati identici. La selezione è GPU-first e automatica: lascia `backend: "auto"`.
- **Porte:** La porta predefinita verificata è la **8227**. Ignora riferimenti alla 8080 se presenti in vecchi snippet.

## 2. Protocollo di Comunicazione (Payload JSON)
L'errore più comune delle AI è la malformazione del campo `instructions`.

**REGOLE MANDATORIE:**
1. **Dizionario, non Lista:** Ogni gate DEVE essere un oggetto: `{"gate": "NOME", "qubits": [indices], "params": {}}`.
2. **Nomi dei Parametri:** 
   - `RX` e `RY` usano il parametro `"theta"`.
   - `RZ` usa il parametro `"phi"`.
   - **MAI** passare l'angolo direttamente nella lista dei qubit.
3. **Esempio Corretto:**
   ```json
   {
     "n_qubits": 2,
     "instructions": [
       {"gate": "H", "qubits": [0]},
       {"gate": "RZ", "qubits": [1], "params": {"phi": 0.785}}
     ]
   }
   ```

## 3. Limiti di Scalabilità e Performance
La scalabilità è limitata dalla memoria: lo statevector occupa **$2^N \times 16$ byte** (complex128). La crescita è esponenziale. Riferimenti su GPU consumer da 8 GB (es. RTX 4070):
- **ZONA VERDE (1-26 Qubit):** Stato ≤ 1 GB. `backend: "auto"` → GPU, risposta rapida.
- **LIMITE GPU 8 GB (28 Qubit):** Stato 4.29 GB, picco ~4.84 GB grazie all'applicazione dei gate a blocchi. È il massimo su questa classe di GPU.
- **OLTRE (≥29 Qubit):** Lo stato (≥8.59 GB) non entra in 8 GB di VRAM: il simulatore intercetta l'out-of-memory e passa **automaticamente a CPU** (più lento, ma senza crash).
- **GPU più grandi:** una GPU con più VRAM alza proporzionalmente il limite (es. ~32 qubit con ~68 GB). Non c'è "scalabilità magica": il costo resta esponenziale.

## 4. Gestione dei Dati (Output)
- **Top States:** Per $N > 16$, il server scansiona solo i primi 65.536 indici. Se lo stato cercato ha un indice alto, non apparirà nella lista `top_states`.
- **Bitstring Final:** Fidati **SEMPRE** di `bitstring_final`. Viene estratto tramite `argmax` globale su tutta la VRAM e rappresenta il risultato corretto anche quando la lista dei top states è vuota.
- **Non richiedere "Tutti i dati":** Per $N > 16$, non tentare di parsare l'intera distribuzione di probabilità. Satureresti la memoria del client.

## 5. Algoritmi Pre-compilati (Best Practice)
Non cercare di decomporre algoritmi complessi in gate elementari se esiste un endpoint dedicato.
- **Grover:** Usa `POST /grover`. È ottimizzato matematicamente in $O(1)$ sul server. Decomporlo a mano a 28 qubit è inefficiente e soggetto a errori di allucinazione dell'oracolo.
- **Bell/GHZ:** Usa `POST /bell` per test rapidi di entanglement.

## 6. Risoluzione Problemi per l'AI
- **Calcoli lenti a ≥29 qubit:** non riscrivere il codice. A quelle dimensioni lo stato non entra in VRAM e il calcolo gira su CPU (lento ma corretto). Riduci il numero di qubit o la profondità del circuito.
- **Unicode Errors:** Evita l'uso di Emoji nei log o negli output se operi in ambiente Windows/PowerShell per prevenire crash di encoding.
