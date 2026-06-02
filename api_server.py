"""
QuantumA Core API Server — REST API per simulatore quantistico.

Espone il motore di simulazione quantistica via HTTP:
- POST /simulate — Esegui un circuito quantistico
- GET  /status   — Stato del server e capacità
- GET  /platforms — Profili hardware disponibili
- POST /grover   — Algoritmo di Grover pre-configurato
- POST /bell     - Stato Bell pre-configurato

Port: 8227 (configurabile via QUANTUM_API_PORT)
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import json
import gc

from quantum_core.compat import configure_stdio, safe_print
from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit
from quantum_core.noise_models import NoiseModelLibrary

try:
    import torch
except Exception:
    torch = None


# ─── Pydantic Models ──────────────────────────────────────────────

class GateInstruction(BaseModel):
    """Istruzione gate per l'API."""
    gate: str = Field(..., description="Nome del gate (H, X, CNOT, ...)")
    qubits: List[int] = Field(..., description="Qubit target/controllo")
    params: Dict[str, float] = Field(default_factory=dict, 
                                      description="Parametri opzionali (theta, phi)")


class CircuitRequest(BaseModel):
    """Richiesta di simulazione circuito."""
    name: Optional[str] = "api_circuit"
    n_qubits: int = Field(..., ge=1, le=30, 
                          description="Numero di qubit")
    instructions: List[GateInstruction] = Field(
        ..., description="Lista di istruzioni gate"
    )
    shots: int = Field(default=1024, ge=1, le=65536,
                       description="Numero di misurazioni")
    mode: str = Field(default="statevector", 
                      description="Modalità: statevector|density_matrix|monte_carlo")
    hardware_profile: str = Field(
        default="superconducting",
        description="Profilo hardware"
    )
    backend: str = Field(
        default="auto",
        description="Backend di esecuzione: auto|cpu|cuda"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seme RNG opzionale per risultati riproducibili"
    )


class GroverRequest(BaseModel):
    """Richiesta per algoritmo di Grover."""
    n_qubits: int = Field(..., ge=2, le=20)
    target_state: Optional[int] = None
    iterations: int = Field(default=1, ge=1)
    shots: int = Field(default=1024, ge=1, le=65536)
    mode: str = "statevector"
    hardware_profile: str = "superconducting"


class BellRequest(BaseModel):
    """Richiesta per stato Bell."""
    n_qubits: int = Field(default=2, ge=2, le=10)
    qubit_a: int = 0
    qubit_b: int = 1
    shots: int = Field(default=1024, ge=1, le=65536)
    mode: str = "statevector"
    hardware_profile: str = "superconducting"


class SimulationResponse(BaseModel):
    """Risposta della simulazione."""
    success: bool
    circuit_name: str
    n_qubits: int
    shots_completed: int
    simulation_time_ms: float
    
    # Risultati principali
    top_states: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Stati più probabili con probabilità"
    )
    
    bitstring_final: Optional[str] = Field(
        default=None,
        description="Bitstring finale candidato estratto dal top state"
    )
    
    probability_distribution: Dict[str, float] = Field(
        default_factory=dict
    )
    
    # Metriche
    circuit_fidelity: float
    estimated_error_rate: float
    gate_count: int
    two_qubit_gates: int
    
    # Stats hardware
    mode: str
    hardware_profile: str


class PlatformInfo(BaseModel):
    """Informazioni su un profilo hardware."""
    name: str
    t1_us: float
    t2_us: float
    fidelity_single: float
    fidelity_two: float
    gate_time_single_ns: float
    gate_time_two_ns: float


class BenchmarkRequest(BaseModel):
    """Richiesta benchmark API."""
    backend: str = Field(default="auto", description="auto|cpu|cuda")
    qubits: List[int] = Field(default_factory=lambda: [8, 10, 12], description="Lista di qubit da testare")
    shots: int = Field(default=1024, ge=1, le=65536)
    repeats: int = Field(default=3, ge=1, le=10)


# ─── FastAPI App ──────────────────────────────────────────────

configure_stdio()

app = FastAPI(
    title="QuantumA Core API",
    description="Quantum chip simulator REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_VALID_PLATFORMS = frozenset({'superconducting', 'trapped_ion', 'silicon_spin', 'neutral_atom'})
_VALID_MODES = frozenset({'statevector', 'density_matrix', 'monte_carlo'})


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check e info."""
    return {
        "service": "QuantumA Core",
        "version": "1.0.0",
        "description": "Simulatore di Chip Quantistico Realistico",
        "endpoints": {
            "POST /simulate": "Esegui circuito quantistico personalizzato",
            "GET /status": "Stato del server",
            "GET /platforms": "Lista profili hardware",
            "POST /grover": "Algoritmo di Grover pre-configurato",
            "POST /bell": "Stato Bell pre-configurato",
        }
    }


def cleanup_quantum_memory():
    """Rilascia buffer temporanei CPU/GPU dopo una simulazione."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def extract_best_bitstring(top_states: List[Dict[str, Any]]) -> Optional[str]:
    if not top_states:
        return None
    best = max(top_states, key=lambda s: float(s.get("probability", 0.0)))
    return best.get("state")


def extract_best_bitstring_from_statevector(statevector: Any, n_qubits: int) -> Optional[str]:
    if statevector is None:
        return None
    try:
        amplitudes = statevector.detach().cpu().numpy() if hasattr(statevector, "detach") else np.asarray(statevector)
        if amplitudes.size == 0:
            return None
        idx = int(np.argmax(np.abs(amplitudes) ** 2))
        return format(idx, f"0{n_qubits}b")
    except Exception:
        return None


def get_gpu_diagnostics():
    cuda_available = bool(torch and torch.cuda.is_available())
    reason = None
    if torch is None:
        reason = "torch_not_installed"
    elif not cuda_available:
        reason = "torch_cpu_build_or_no_cuda_runtime"

    return {
        "torch_available": torch is not None,
        "cuda_available": cuda_available,
        "torch_version": getattr(torch, "__version__", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "fallback_reason": reason,
    }


@app.get("/status")
async def get_status():
    """Stato del server."""
    return {
        "status": "running",
        "max_qubits_statevector": 30,
        "max_qubits_density_matrix": 18,
        "supported_modes": ["statevector", "density_matrix", "monte_carlo"],
        "supported_platforms": [
            "superconducting", 
            "trapped_ion", 
            "silicon_spin", 
            "neutral_atom"
        ],
        "gpu": get_gpu_diagnostics(),
        "documentation": "QuantumA Core v1.0.0 — Simulatore quantistico realistico"
    }


@app.get("/status/gpu")
async def get_gpu_status():
    """Diagnostica GPU/CUDA."""
    return get_gpu_diagnostics()


@app.get("/platforms")
async def get_platforms():
    """Lista dei profili hardware disponibili."""
    platforms = [
        ("superconducting", NoiseModelLibrary.superconducting_transmon()),
        ("trapped_ion", NoiseModelLibrary.trapped_ion()),
        ("silicon_spin", NoiseModelLibrary.silicon_spin()),
        ("neutral_atom", NoiseModelLibrary.neutral_atom()),
    ]
    
    return [
        PlatformInfo(
            name=name,
            t1_us=p.t1,
            t2_us=p.t2,
            fidelity_single=p.fidelity_single,
            fidelity_two=p.fidelity_two,
            gate_time_single_ns=p.t_gate_single * 1000,
            gate_time_two_ns=p.t_gate_two * 1000,
        ).dict()
        for name, p in platforms
    ]


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """Benchmark breve del simulatore statevector."""
    def build_chain(n_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        return qc

    results = []
    for n in request.qubits:
        if n < 1 or n > 30:
            raise HTTPException(400, f"Invalid qubit count in benchmark: {n}")
        qc = build_chain(n)
        timings = []
        backend_used = None
        for _ in range(request.repeats):
            sim = QuantumSimulator(
                n_qubits=n,
                mode=SimulationMode.STATEVECTOR,
                hardware_profile="superconducting",
                backend=request.backend,
            )
            start = __import__("time").perf_counter()
            sim.run_circuit(qc, shots=request.shots)
            elapsed = (__import__("time").perf_counter() - start) * 1000
            timings.append(elapsed)
            backend_used = sim.backend
        results.append({
            "n_qubits": n,
            "avg_ms": round(sum(timings) / len(timings), 3),
            "min_ms": round(min(timings), 3),
            "max_ms": round(max(timings), 3),
            "backend_used": backend_used,
            "requested_backend": request.backend,
            "fallback_to_cpu": backend_used != request.backend,
        })

    return {
        "success": True,
        "requested_backend": request.backend,
        "results": results,
    }


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: CircuitRequest):
    """
    Esegui un circuito quantistico.
    
    - Crea un simulatore con il numero di qubit richiesto
    - Costruisce il circuito dalle istruzioni
    - Esegue la simulazione nella modalità specificata
    - Ritorna i risultati con distribuzione di probabilità
    
    Esempio (stato Bell):
    ```json
    {
        "n_qubits": 2,
        "instructions": [
            {"gate": "H", "qubits": [0]},
            {"gate": "CNOT", "qubits": [0, 1]}
        ],
        "shots": 1024,
        "mode": "statevector"
    }
    ```
    """
    if request.mode not in _VALID_MODES:
        raise HTTPException(400, f"Invalid mode '{request.mode}'. Valid: {sorted(_VALID_MODES)}")
    if request.hardware_profile not in _VALID_PLATFORMS:
        raise HTTPException(400, f"Invalid hardware_profile '{request.hardware_profile}'. Valid: {sorted(_VALID_PLATFORMS)}")
    
    simulator = None
    circuit = None
    result = None
    try:
        # Crea simulatore
        sim_mode = SimulationMode(request.mode)
        
        # Limita qubit per modalità density matrix (memory intensive)
        if request.n_qubits > 18 and sim_mode == SimulationMode.DENSITY_MATRIX:
            raise HTTPException(
                400, 
                f"Density matrix limited to ~18 qubits. Got {request.n_qubits}. "
                "Use statevector mode for more qubits."
            )
        
        simulator = QuantumSimulator(
            n_qubits=request.n_qubits,
            mode=sim_mode,
            hardware_profile=request.hardware_profile,
            backend=request.backend,
            seed=request.seed,
        )

        # Costruisci circuito
        circuit = QuantumCircuit(n_qubits=request.n_qubits, name=request.name)
        
        for instr in request.instructions:
            gate_name = instr.gate.upper()
            
            if len(instr.qubits) == 1:
                single_gate_map = {
                    'H': circuit.h,
                    'X': circuit.x,
                    'Y': circuit.y,
                    'Z': circuit.z,
                    'S': circuit.s,
                    'T': circuit.t,
                    'RX': circuit.rx,
                    'RY': circuit.ry,
                    'RZ': circuit.rz,
                }
                if gate_name not in single_gate_map:
                    raise HTTPException(400, f"Unsupported single-qubit gate: {gate_name}")
                single_gate_map[gate_name](instr.qubits[0], **(instr.params or {}))

            elif len(instr.qubits) == 2:
                two_qubit_map = {
                    'CNOT': circuit.cx,
                    'CX': circuit.cx,
                    'CZ': circuit.cz,
                    'SWAP': circuit.swap,
                    'ISWAP': circuit.iswap,
                    'RZZ': circuit.rzz,
                    'CRZ': circuit.crz,
                    'CPHASE': circuit.cphase,
                }
                if gate_name not in two_qubit_map:
                    raise HTTPException(400, f"Unsupported two-qubit gate: {gate_name}")
                two_qubit_map[gate_name](instr.qubits[0], instr.qubits[1], **(instr.params or {}))
            elif len(instr.qubits) == 3:
                three_qubit_map = {
                    'CCNOT': circuit.ccnot,
                    'TOFFOLI': circuit.ccnot,
                }
                if gate_name not in three_qubit_map:
                    raise HTTPException(400, f"Unsupported three-qubit gate: {gate_name}")
                three_qubit_map[gate_name](instr.qubits[0], instr.qubits[1], instr.qubits[2])
            else:
                raise HTTPException(400, f"Unsupported gate arity: {len(instr.qubits)}")
        
        # Esegui simulazione
        result = simulator.run_circuit(circuit, shots=request.shots)
        
        # Costruisci risposta
        top_states = []
        for state, prob in result.most_likely_states(10):
            top_states.append({
                "state": state,
                "probability": round(prob, 6),
            })
        
        return SimulationResponse(
            success=True,
            circuit_name=request.name,
            n_qubits=request.n_qubits,
            shots_completed=result.shots_completed,
            simulation_time_ms=round(result.simulation_time_ms, 3),
            top_states=top_states,
            bitstring_final=extract_best_bitstring_from_statevector(result.final_statevector, request.n_qubits) or extract_best_bitstring(top_states),
            probability_distribution={k: round(v, 8) for k, v in result.probability_distribution.items()},
            circuit_fidelity=round(result.circuit_fidelity, 6),
            estimated_error_rate=round(result.estimated_error_rate, 6),
            gate_count=result.gate_count,
            two_qubit_gates=circuit.two_qubit_gates,
            mode=request.mode,
            hardware_profile=request.hardware_profile,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Simulation error: {str(e)}")
    finally:
        del result
        del circuit
        del simulator
        cleanup_quantum_memory()


@app.post("/grover", response_model=SimulationResponse)
async def run_grover(request: GroverRequest):
    """
    Esegue l'algoritmo di Grover.
    
    - Inizializza sovrapposizione uniforme con H^⊗n
    - Applica oracle + diffusion operator per n_iterations volte
    - Misura e ritorna distribuzione di probabilità
    
    L'oracle marca lo stato target (o |11...1⟩ se non specificato).
    """
    simulator = None
    result = None
    if request.n_qubits < 2:
        raise HTTPException(400, "Grover requires at least 2 qubits")
    if request.mode != "statevector":
        raise HTTPException(400, "Grover endpoint requires statevector mode")
    if request.hardware_profile not in _VALID_PLATFORMS:
        raise HTTPException(400, f"Invalid hardware_profile '{request.hardware_profile}'. Valid: {sorted(_VALID_PLATFORMS)}")

    try:
        sim_mode = SimulationMode(request.mode)
        
        simulator = QuantumSimulator(
            n_qubits=request.n_qubits,
            mode=sim_mode,
            hardware_profile=request.hardware_profile,
        )
        
        result = simulator.run_grover(
            n_iterations=request.iterations,
            target_state=request.target_state,
            shots=request.shots,
        )
        
        top_states = []
        for state, prob in result.most_likely_states(10):
            top_states.append({
                "state": state,
                "probability": round(prob, 6),
            })
        
        return SimulationResponse(
            success=True,
            circuit_name="grover",
            n_qubits=request.n_qubits,
            shots_completed=result.shots_completed,
            simulation_time_ms=round(result.simulation_time_ms, 3),
            top_states=top_states,
            bitstring_final=extract_best_bitstring_from_statevector(result.final_statevector, request.n_qubits) or extract_best_bitstring(top_states),
            probability_distribution={k: round(v, 8) for k, v in result.probability_distribution.items()},
            circuit_fidelity=round(result.circuit_fidelity, 6),
            estimated_error_rate=round(result.estimated_error_rate, 6),
            gate_count=result.gate_count,
            two_qubit_gates=0,  # Calcolato dal simulatore interno
            mode=request.mode,
            hardware_profile=request.hardware_profile,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Grover simulation error: {str(e)}")
    finally:
        del result
        del simulator
        cleanup_quantum_memory()


@app.post("/bell", response_model=SimulationResponse)
async def run_bell(request: BellRequest):
    """
    Esegue un circuito stato Bell.
    
    Crea (|00⟩ + |11⟩)/√2 sui qubit specificati.
    """
    simulator = None
    result = None
    if request.mode not in _VALID_MODES:
        raise HTTPException(400, f"Invalid mode '{request.mode}'. Valid: {sorted(_VALID_MODES)}")
    if request.hardware_profile not in _VALID_PLATFORMS:
        raise HTTPException(400, f"Invalid hardware_profile '{request.hardware_profile}'. Valid: {sorted(_VALID_PLATFORMS)}")
    try:
        sim_mode = SimulationMode(request.mode)

        simulator = QuantumSimulator(
            n_qubits=request.n_qubits,
            mode=sim_mode,
            hardware_profile=request.hardware_profile,
        )

        result = simulator.run_bell_state(
            qubit_a=request.qubit_a,
            qubit_b=request.qubit_b,
            shots=request.shots,
        )
        
        top_states = []
        for state, prob in result.most_likely_states(10):
            top_states.append({
                "state": state,
                "probability": round(prob, 6),
            })
        
        return SimulationResponse(
            success=True,
            circuit_name="bell",
            n_qubits=request.n_qubits,
            shots_completed=result.shots_completed,
            simulation_time_ms=round(result.simulation_time_ms, 3),
            top_states=top_states,
            bitstring_final=extract_best_bitstring_from_statevector(result.final_statevector, request.n_qubits) or extract_best_bitstring(top_states),
            probability_distribution={k: round(v, 8) for k, v in result.probability_distribution.items()},
            circuit_fidelity=round(result.circuit_fidelity, 6),
            estimated_error_rate=round(result.estimated_error_rate, 6),
            gate_count=result.gate_count,
            two_qubit_gates=1,
            mode=request.mode,
            hardware_profile=request.hardware_profile,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Bell simulation error: {str(e)}")
    finally:
        del result
        del simulator
        cleanup_quantum_memory()


# ─── Main ──────────────────────────────────────────────

def start_server(host="127.0.0.1", port=8227):
    """Avvia il server API programmaticamente."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    port = int(os.environ.get("QUANTUM_API_PORT", 8227))
    host = os.environ.get("QUANTUM_API_HOST", "127.0.0.1")
    
    banner = (
        "QuantumA Core API Server v1.0.0\n"
        "Simulatore di Chip Quantistico Realistico\n"
        f"http://{host}:{port} | /docs | /status"
    )
    safe_print(banner)

    start_server(host=host, port=port)
