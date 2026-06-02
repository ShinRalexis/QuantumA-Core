# QuantumA Core — Guida all'API REST

Guida all'uso del simulatore QuantumA Core tramite la sua API REST (FastAPI).

## Avvio del server

```bash
python api_server.py
```

Il server è disponibile su `http://127.0.0.1:8227`. In Docker:

```bash
docker compose up -d --build
```

> Il backend GPU si attiva solo se il runtime vede CUDA (`torch.cuda.is_available()`).
> In caso contrario il server resta operativo in modalità CPU.

---

## Documentazione interattiva

QuantumA Core espone un'interfaccia **Swagger/OpenAPI**. Ad avvio del server, visita:
**[http://127.0.0.1:8227/docs](http://127.0.0.1:8227/docs)** — qui puoi provare ogni
endpoint dal browser.

---

## Endpoint

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| `GET` | `/` | info di base e lista endpoint |
| `GET` | `/status` | stato del server, modalità e diagnostica GPU |
| `GET` | `/status/gpu` | solo diagnostica GPU/CUDA |
| `GET` | `/platforms` | profili hardware di rumore disponibili |
| `POST` | `/simulate` | esegue un circuito personalizzato |
| `POST` | `/bell` | stato di Bell pre-configurato |
| `POST` | `/grover` | algoritmo di Grover pre-configurato |
| `POST` | `/benchmark` | benchmark breve dello statevector |

---

## `POST /simulate`

Corpo della richiesta:

```json
{
  "n_qubits": 3,
  "instructions": [
    {"gate": "h", "qubits": [0]},
    {"gate": "cx", "qubits": [0, 1]},
    {"gate": "ry", "qubits": [2], "params": {"theta": 0.78}}
  ],
  "shots": 1024,
  "mode": "statevector",
  "backend": "auto",
  "hardware_profile": "superconducting",
  "seed": 42
}
```

| Campo | Default | Note |
| --- | --- | --- |
| `n_qubits` | — | obbligatorio (1–30) |
| `instructions` | — | obbligatorio; ogni gate ha `gate`, `qubits`, e `params` opzionale |
| `shots` | 1024 | 1–65536 |
| `mode` | `statevector` | `statevector` / `density_matrix` / `monte_carlo` |
| `backend` | `auto` | `auto` / `cpu` / `cuda` |
| `hardware_profile` | `superconducting` | profilo di rumore |
| `seed` | `null` | seme RNG per risultati riproducibili |

**Gate supportati** (case-insensitive): `H, X, Y, Z, S, T, RX, RY, RZ` (1 qubit);
`CNOT/CX, CZ, SWAP, ISWAP, RZZ, CRZ, CPHASE` (2 qubit); `CCNOT/TOFFOLI` (3 qubit).

La risposta include `top_states`, `probability_distribution`, `bitstring_final`,
`circuit_fidelity`, `gate_count` e i tempi di esecuzione.

---

## Esempi di integrazione

### Python (requests)

```python
import requests

payload = {
    "n_qubits": 2,
    "instructions": [
        {"gate": "h", "qubits": [0]},
        {"gate": "cx", "qubits": [0, 1]},
    ],
    "shots": 1000,
    "backend": "auto",
}
r = requests.post("http://127.0.0.1:8227/simulate", json=payload)
print(r.json()["top_states"])
```

### JavaScript (fetch)

```javascript
const res = await fetch('http://127.0.0.1:8227/status');
const data = await res.json();
console.log(`GPU attiva: ${data.gpu.cuda_available}`);
```

### PowerShell

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8227/status" -Method Get | ConvertTo-Json
```

---

## Suggerimenti

- **Lascia `backend: "auto"`**: la selezione è GPU-first e gestisce da sola la
  scelta GPU/CPU in base alla VRAM, con fallback automatico su CPU.
- **Memoria**: lo stato occupa `2ⁿ × 16 byte`. A 26 qubit ≈ 1.07 GB, a 28 qubit
  ≈ 4.29 GB (picco ~4.84 GB su GPU grazie all'applicazione dei gate a blocchi).
  Da 29 qubit in su lo stato non entra in 8 GB di VRAM e si passa a CPU.
- **Riproducibilità**: passa `seed` per ottenere sempre lo stesso risultato.

> Nota di sicurezza: il server abilita CORS per qualsiasi origine ed è senza
> autenticazione. È pensato per uso locale/fidato. Non esporlo su Internet senza
> aggiungere autenticazione e restrizioni di rete.
