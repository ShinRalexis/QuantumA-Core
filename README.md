<p align="center">
  <img src="Logo.png" alt="QuantumA Core" width="220">
</p>

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="#italiano">Italiano</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/quantuma-core/"><img src="https://img.shields.io/pypi/v/quantuma-core" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://github.com/ShinRalexis/QuantumA-Core/actions/workflows/tests.yml/badge.svg" alt="Tests">
  <img src="https://img.shields.io/badge/GPU-CUDA%20(optional)-76b900" alt="CUDA optional">
</p>

# QuantumA Core

**Local, GPU-first, REST API, zero cloud-vendor lock-in.**

A quantum circuit simulator that runs on classical hardware, with optional GPU
(CUDA) acceleration and a REST API. It lets you build, run, and analyze quantum
circuits with realistic noise models, automatically choosing between GPU and CPU
backends.

**It is not a quantum computer.** It is a tool to design and verify quantum
algorithms before running them on real hardware, to study the effect of noise on
NISQ circuits, and to integrate quantum simulation into other applications over
HTTP.

> Status: stable, 35/35 functional tests passing. GPU-first backend selection:
> up to 28 qubits on an 8 GB GPU, automatic CPU fallback beyond.

---

## Features

- **Three simulation modes**: statevector (pure state), density matrix (mixed
  state with noise), Monte Carlo (stochastic trajectories).
- **GPU-first backend**: uses the GPU (PyTorch/CUDA) whenever available, applying
  gates *in chunks* to keep VRAM low. Automatic, crash-free CPU (NumPy) fallback.
- **Realistic noise**: Kraus operators for T1/T2, depolarizing and correlated
  noise, with four hardware profiles (superconducting, trapped ion, silicon spin,
  neutral atom).
- **REST API** (FastAPI) on port 8227, documented at `/docs`.
- **Reproducibility**: `seed` parameter for deterministic results.
- **Real quantum chemistry example**: VQE for the H₂ molecule.

## Requirements

- Python 3.10+
- NumPy (required)
- PyTorch with CUDA (optional, for GPU acceleration)
- FastAPI + Uvicorn (for the REST API)

Install as a library from PyPI:

```bash
pip install quantuma-core          # core (NumPy only)
pip install "quantuma-core[gpu]"   # + PyTorch (GPU acceleration)
pip install "quantuma-core[api]"   # + FastAPI REST server
```

Or, from a clone of the repository:

```bash
pip install -r requirements.txt
```

> For the CUDA build of PyTorch, install `torch` separately following the
> instructions at <https://pytorch.org> (the CUDA index is platform-specific).

---

## Quick start

### Option A: Docker (recommended)

```bash
docker compose up -d --build
```

The API is available at `http://localhost:8227` (docs at `/docs`). The GPU
backend activates only if the container sees CUDA; otherwise it stays on CPU.

### Option B: Local

```bash
python run_check.py      # verification suite (35 tests)
python api_server.py     # start the API on port 8227
```

---

## Examples

### 1. Bell state in Python

```python
from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)                     # superposition + entanglement
result = sim.run_circuit(qc, shots=1000)
print(result.most_likely_states(2))
```

Expected output:

```text
[('00', 0.5), ('11', 0.5)]
```

### 2. Same circuit via REST API

```bash
curl -X POST http://localhost:8227/simulate \
  -H "Content-Type: application/json" \
  -d '{"n_qubits":2,"instructions":[{"gate":"h","qubits":[0]},{"gate":"cx","qubits":[0,1]}],"shots":1024}'
```

The response includes `top_states`, `probability_distribution`,
`bitstring_final`, `circuit_fidelity`, and execution metrics.

### 3. Quantum chemistry: VQE for H₂

```bash
python molecular_chemistry_test.py
```

Computes the ground-state energy of H₂ from first principles (STO-3G integrals →
Hartree-Fock → 4-qubit Jordan-Wigner Hamiltonian → VQE on the simulator). Output
excerpt:

```text
Hartree-Fock    : -1.116714 Ha   (expected ~ -1.1167)
VQE (QuantumA)  : -1.137276 Ha
FCI (exact)     : -1.137276 Ha   (expected ~ -1.1373)
Correlation energy (FCI-HF): -20.562 mHa = -12.90 kcal/mol
VQE vs FCI error: 0.236 microHartree
```

VQE reproduces the exact (FCI) energy to 0.24 microHartree, and the dissociation
curve recovers the correct equilibrium bond length (0.741 Å). Every value is
validated against the literature via checkpoints in the code.

---

## Tests

```bash
python run_check.py
```

Runs 35 functional tests: module imports, gate unitarity, statevector engine
(CPU), density matrix, circuit builder, noise models, and all simulator modes.
All must report `[OK]`.

```text
============================================================
  CHECK COMPLETE:  35 passed  |  0 failed
============================================================
```

---

## Use cases

1. **Teaching and learning**: build and inspect circuits, Bell/GHZ states, known
   algorithms (Grover, QFT), with ASCII rendering and probabilities.
2. **NISQ algorithm prototyping**: verify an algorithm in ideal mode
   (statevector) and then under realistic noise before porting it to real
   hardware.
3. **Noise studies**: compare ideal execution with density matrix / Monte Carlo
   across the four hardware profiles.
4. **Quantum chemistry**: VQE for molecular energies (see the H₂ example).
5. **Integration into other applications**: any software can use quantum
   simulation over HTTP, without depending on a specific vendor's ecosystem.

---

## Hardware scalability

The state vector grows as `2ⁿ × 16 bytes` (complex128): exponential in the number
of qubits. Backend selection is GPU-first.

| Qubits | State | Backend | Notes |
| --- | --- | --- | --- |
| ≤ 26 | ≤ 1 GB | GPU | fast |
| 28 | 4.29 GB | GPU | chunked gate application, peak ~4.84 GB on 8 GB |
| ≥ 29 | ≥ 8.59 GB | CPU | state doesn't fit in 8 GB VRAM → automatic fallback |

A GPU with more VRAM raises the limit proportionally. Forcing `backend="cuda"`
skips the automatic check; any out-of-memory is intercepted with a CPU fallback
without crashing.

*Measured on RTX 4070 Laptop (8 GB). At 28 qubits the GPU backend is ~4× faster
than CPU on the same circuit.*

---

## REST API (main endpoints)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/status` | server status and GPU diagnostics |
| `POST` | `/simulate` | run a custom circuit |
| `POST` | `/bell` | pre-configured Bell state |
| `POST` | `/grover` | pre-configured Grover's algorithm |
| `GET` | `/platforms` | available noise hardware profiles |
| `GET` | `/docs` | interactive OpenAPI documentation |

---

## Project structure

```text
quantum_core/
  statevector.py      # pure-state engine (CPU, NumPy/einsum)
  gpu_statevector.py  # pure-state engine (GPU, PyTorch/CUDA, chunked einsum)
  density_matrix.py   # mixed states and noise channels
  gates.py            # gate library (single source, CPU and GPU)
  noise_models.py     # Kraus operators, hardware profiles
  circuit.py          # circuit builder, optimization, serialization
  simulator.py        # orchestrator, backend selection, VQE/Grover/Bell
api_server.py             # FastAPI server (port 8227)
run_check.py              # verification suite (35 tests)
molecular_chemistry_test.py  # real example: VQE for H₂
```

---

## Projects using QuantumA Core

- **Silly Quantum**: an extension for [SillyTavern](https://github.com/SillyTavern/SillyTavern)
  that uses the entropy of real quantum circuits (via the QuantumA Core API) to
  modulate characters' emotional state in roleplay. An example of downstream
  integration over REST.

---

## Known limitations

- The GPU backend applies 1–2 qubit gates; 3-qubit gates (Toffoli) run on CPU.
- Grover's algorithm is implemented analytically directly on the statevector, not
  as a gate sequence.
- Density matrix mode is limited to ~16–18 qubits (the density matrix has
  `2^(2n)` elements).
- The chemistry example covers H₂ in a minimal basis (4 qubits); larger molecules
  would require a quantum chemistry package to generate the integrals.

---

## Security

> ⚠️ The API server enables **CORS for any origin** (`allow_origins=["*"]`) and
> has **no authentication**. It is designed for local or trusted-network use.
> **Do not expose it directly to the Internet** without adding authentication,
> rate limiting, and network restrictions in front of it.

---

## License

QuantumA Core is released under the **MIT License**, free and open source. You
are free to use, modify, and distribute it (including commercially), as long as
the copyright notice is retained. See the [LICENSE](LICENSE) file.

Author: **MetaDarko** · Contact: <MetaDarko@pm.me>

---

## Support the project

QuantumA Core is developed by **MetaDarko**. If you find the project useful, you
can support its development with a donation:

[![Liberapay](https://img.shields.io/badge/Liberapay-Support-yellow)](https://liberapay.com/MetaDarko)

Support page: [liberapay.com/MetaDarko](https://liberapay.com/MetaDarko)

<br>

---
---

<a id="italiano"></a>

<p align="center">
  <a href="#quantuma-core">English</a> &nbsp;|&nbsp; <b>Italiano</b>
</p>

# QuantumA Core (Italiano)

**Locale, GPU-first, REST API, zero dipendenza da cloud vendor.**

Simulatore di circuiti quantistici su hardware classico, con accelerazione GPU
(CUDA) opzionale e API REST. Permette di costruire, eseguire e analizzare
circuiti quantistici con modelli di rumore realistici, scegliendo
automaticamente tra backend GPU e CPU.

**Non è un computer quantistico.** È uno strumento per progettare e verificare
algoritmi quantistici prima di eseguirli su hardware reale, studiare l'effetto
del rumore su circuiti NISQ, e integrare la simulazione quantistica in altre
applicazioni tramite HTTP.

> Stato: stabile, 35/35 test funzionali passati. Selezione backend GPU-first:
> fino a 28 qubit su una GPU da 8 GB, fallback automatico su CPU oltre.

---

## Caratteristiche

- **Tre modalità di simulazione**: statevector (stato puro), density matrix
  (stato misto con rumore), Monte Carlo (traiettorie stocastiche).
- **Backend GPU-first**: usa la GPU (PyTorch/CUDA) ogni volta che è disponibile,
  con applicazione dei gate *a blocchi* per contenere la VRAM. Fallback CPU
  (NumPy) automatico e senza crash.
- **Rumore realistico**: operatori di Kraus per T1/T2, depolarizing e rumore
  correlato, con quattro profili hardware (superconduttore, trappola ionica,
  spin nel silicio, atomo neutro).
- **API REST** (FastAPI) sulla porta 8227, documentata su `/docs`.
- **Riproducibilità**: parametro `seed` per risultati deterministici.
- **Esempio di chimica quantistica reale**: VQE per la molecola H₂.

## Requisiti

- Python 3.10+
- NumPy (obbligatorio)
- PyTorch con CUDA (opzionale, per l'accelerazione GPU)
- FastAPI + Uvicorn (per l'API REST)

Installazione come libreria da PyPI:

```bash
pip install quantuma-core          # core (solo NumPy)
pip install "quantuma-core[gpu]"   # + PyTorch (accelerazione GPU)
pip install "quantuma-core[api]"   # + server REST FastAPI
```

Oppure, da un clone del repository:

```bash
pip install -r requirements.txt
```

> Per la build CUDA di PyTorch, installa `torch` separatamente seguendo le
> istruzioni su <https://pytorch.org> (l'indice CUDA dipende dalla piattaforma).

---

## Avvio rapido

### Opzione A: Docker (consigliata)

```bash
docker compose up -d --build
```

L'API è disponibile su `http://localhost:8227` (documentazione su `/docs`).
Il backend GPU si attiva solo se il container vede CUDA; altrimenti resta su CPU.

### Opzione B: Locale

```bash
python run_check.py      # suite di verifica (35 test)
python api_server.py     # avvia l'API su porta 8227
```

---

## Esempi

### 1. Stato di Bell in Python

```python
from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)                     # sovrapposizione + entanglement
result = sim.run_circuit(qc, shots=1000)
print(result.most_likely_states(2))
```

Output atteso:

```text
[('00', 0.5), ('11', 0.5)]
```

### 2. Stesso circuito via API REST

```bash
curl -X POST http://localhost:8227/simulate \
  -H "Content-Type: application/json" \
  -d '{"n_qubits":2,"instructions":[{"gate":"h","qubits":[0]},{"gate":"cx","qubits":[0,1]}],"shots":1024}'
```

La risposta include `top_states`, `probability_distribution`, `bitstring_final`,
`circuit_fidelity` e le metriche di esecuzione.

### 3. Chimica quantistica: VQE per H₂

```bash
python molecular_chemistry_test.py
```

Calcola l'energia dello stato fondamentale di H₂ da principi primi (integrali
STO-3G → Hartree-Fock → Hamiltoniano Jordan-Wigner a 4 qubit → VQE sul
simulatore). Estratto dell'output:

```text
Hartree-Fock    : -1.116714 Ha   (atteso ~ -1.1167)
VQE (QuantumA)  : -1.137276 Ha
FCI (esatto)    : -1.137276 Ha   (atteso ~ -1.1373)
Energia di correlazione (FCI-HF): -20.562 mHa = -12.90 kcal/mol
Errore VQE vs FCI: 0.236 microHartree
```

Il VQE riproduce l'energia esatta (FCI) a 0.24 microHartree, e la curva di
dissociazione restituisce la lunghezza di legame d'equilibrio corretta (0.741 Å).
Ogni valore è validato contro la letteratura tramite checkpoint nel codice.

---

## Test

```bash
python run_check.py
```

Esegue 35 test funzionali: import dei moduli, unitarietà dei gate, motore
statevector (CPU), density matrix, circuit builder, modelli di rumore e tutte
le modalità del simulatore. Tutti devono risultare `[OK]`.

```text
============================================================
  CHECK COMPLETE:  35 passed  |  0 failed
============================================================
```

---

## Casi d'uso

1. **Didattica e apprendimento**: costruire e ispezionare circuiti, stati di
   Bell/GHZ, algoritmi noti (Grover, QFT), con rendering ASCII e probabilità.
2. **Prototipazione di algoritmi NISQ**: verificare un algoritmo in modalità
   ideale (statevector) e poi sotto rumore realistico prima di portarlo su
   hardware reale.
3. **Studio del rumore**: confrontare l'esecuzione ideale con density matrix /
   Monte Carlo sui quattro profili hardware.
4. **Chimica quantistica**: VQE per energie molecolari (vedi l'esempio H₂).
5. **Integrazione in altre applicazioni**: qualsiasi software può usare la
   simulazione quantistica via HTTP, senza dipendere dall'ecosistema di un
   vendor specifico.

---

## Scalabilità hardware

Il vettore di stato cresce come `2ⁿ × 16 byte` (complex128): esponenziale nel
numero di qubit. La selezione del backend è GPU-first.

| Qubit | Stato | Backend | Note |
| --- | --- | --- | --- |
| ≤ 26 | ≤ 1 GB | GPU | veloce |
| 28 | 4.29 GB | GPU | applicazione gate a blocchi, picco ~4.84 GB su 8 GB |
| ≥ 29 | ≥ 8.59 GB | CPU | lo stato non entra in 8 GB di VRAM → fallback automatico |

Una GPU con più VRAM alza proporzionalmente il limite. Forzando
`backend="cuda"` si salta il controllo automatico; un eventuale out-of-memory
viene intercettato con fallback su CPU senza crashare.

*Misure su RTX 4070 Laptop (8 GB). A 28 qubit il backend GPU è ~4× più veloce
della CPU sullo stesso circuito.*

---

## API REST (principali endpoint)

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| `GET` | `/status` | stato del server e diagnostica GPU |
| `POST` | `/simulate` | esegue un circuito personalizzato |
| `POST` | `/bell` | stato di Bell pre-configurato |
| `POST` | `/grover` | algoritmo di Grover pre-configurato |
| `GET` | `/platforms` | profili hardware di rumore disponibili |
| `GET` | `/docs` | documentazione interattiva OpenAPI |

---

## Struttura del progetto

```text
quantum_core/
  statevector.py      # motore stato puro (CPU, NumPy/einsum)
  gpu_statevector.py  # motore stato puro (GPU, PyTorch/CUDA, einsum a blocchi)
  density_matrix.py   # stati misti e canali di rumore
  gates.py            # libreria gate (sorgente unica, CPU e GPU)
  noise_models.py     # operatori di Kraus, profili hardware
  circuit.py          # builder di circuiti, ottimizzazione, serializzazione
  simulator.py        # orchestratore, selezione backend, VQE/Grover/Bell
api_server.py             # server FastAPI (porta 8227)
run_check.py              # suite di verifica (35 test)
molecular_chemistry_test.py  # esempio reale: VQE per H₂
```

---

## Progetti che usano QuantumA Core

- **Silly Quantum**: estensione per [SillyTavern](https://github.com/SillyTavern/SillyTavern)
  che usa l'entropia di circuiti quantistici reali (via l'API di QuantumA Core)
  per modulare lo stato emotivo dei personaggi nel roleplay. Esempio di
  integrazione downstream tramite REST.

---

## Limiti noti

- Il backend GPU applica gate a 1–2 qubit; i gate a 3 qubit (Toffoli) girano su
  CPU.
- L'algoritmo di Grover è implementato in forma analitica diretta sullo
  statevector, non come sequenza di gate.
- La modalità density matrix è limitata a ~16–18 qubit (la matrice densità
  occupa `2^(2n)` elementi).
- L'esempio di chimica copre H₂ in base minimale (4 qubit); molecole più grandi
  richiederebbero un pacchetto di chimica quantistica per generare gli integrali.

---

## Sicurezza

> ⚠️ Il server API abilita **CORS per qualsiasi origine** (`allow_origins=["*"]`)
> ed è **senza autenticazione**. È pensato per uso locale o su rete fidata.
> **Non esporlo direttamente su Internet** senza aggiungere davanti
> autenticazione, rate limiting e restrizioni di rete.

---

## Licenza

QuantumA Core è rilasciato con licenza **MIT**, libera e open source. Sei libero
di usarlo, modificarlo e distribuirlo (anche commercialmente), purché venga
mantenuta la nota di copyright. Vedi il file [LICENSE](LICENSE).

Autore: **MetaDarko** · Contatto: <MetaDarko@pm.me>

---

## Sostieni il progetto

QuantumA Core è sviluppato da **MetaDarko**. Se il progetto ti è utile, puoi
sostenerne lo sviluppo con una donazione:

[![Liberapay](https://img.shields.io/badge/Liberapay-Sostieni-yellow)](https://liberapay.com/MetaDarko)

Pagina di supporto: [liberapay.com/MetaDarko](https://liberapay.com/MetaDarko)
