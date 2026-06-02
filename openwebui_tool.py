"""
title: QuantumA Core Simulator Tool
author: MetaDarko
author_url: https://liberapay.com/MetaDarko
version: 2.0.0
"""

import json
import requests


class Tools:
    def __init__(self):
        # Comunicazione Open WebUI (Docker) -> host che esegue QuantumA Core.
        # Porta predefinita di QuantumA Core: 8227.
        self.api_url = "http://host.docker.internal:8227"

    def _clean_json(self, circuit_json: str):
        clean_json = circuit_json.strip().replace("```json", "").replace("```", "")
        circuit = json.loads(clean_json)

        if isinstance(circuit, dict):
            if "instructions" in circuit:
                circuit = circuit["instructions"]
            elif "gates" in circuit:
                circuit = circuit["gates"]
            elif "circuit" in circuit:
                circuit = circuit["circuit"]
            else:
                raise ValueError(
                    "Devi restituire SOLO un Array JSON di istruzioni o un oggetto con instructions/gates/circuit."
                )

        if not isinstance(circuit, list):
            raise ValueError("Il formato generato non è un Array valido.")

        return circuit

    def _format_states(self, result: dict) -> str:
        output = ""
        top_states = result.get("top_states") or []
        probability_distribution = result.get("probability_distribution") or {}

        states = []
        if top_states:
            states = [
                {
                    "state": s.get("state", "?"),
                    "probability": float(s.get("probability", 0.0)),
                }
                for s in top_states
            ]
        elif probability_distribution:
            states = [
                {"state": state, "probability": float(prob)}
                for state, prob in probability_distribution.items()
            ]
            states.sort(key=lambda x: x["probability"], reverse=True)

        if states:
            output += "- **Distribuzione delle probabilità:**\n"
            for s in states[:10]:
                prob = float(s.get("probability", 0.0))
                bar = "█" * max(1, int(prob * 20)) if prob > 0 else ""
                output += f"  * `|{s.get('state', '?')}>` : {prob:.1%} {bar}\n"
            if len(states) > 10:
                output += f"  * ... altri {len(states) - 10} stati non mostrati\n"
        else:
            output += "- **Distribuzione delle probabilità:** non disponibile (output compatto)\n"

        bitstring_final = result.get("bitstring_final")
        if bitstring_final:
            output += f"\n- **Bitstring finale candidato:** `{bitstring_final}`\n"

        if not top_states and probability_distribution:
            output += "\n- **Nota:** top_states non presente; ricostruita la distribuzione da probability_distribution.\n"

        return output

    def _format_simulation_result(self, result: dict, n_qubits: int) -> str:
        output = f"### 🌌 Risultati Simulazione Quantistica ({n_qubits} qubit)\n"
        output += f"- **Tempo di calcolo:** {result.get('simulation_time_ms', 0):.2f} ms\n"
        output += f"- **Shots completati:** {result.get('shots_completed', 0)}\n"
        output += f"- **Fidelity circuito:** {result.get('circuit_fidelity', 0):.6f}\n"
        output += f"- **Error rate stimato:** {result.get('estimated_error_rate', 0):.6f}\n"
        output += self._format_states(result)
        output += f"\n- **Gate count:** {result.get('gate_count', 0)}\n"
        output += f"- **Two-qubit gates:** {result.get('two_qubit_gates', 0)}\n"
        output += f"- **Modalità:** `{result.get('mode', 'statevector')}`\n"
        output += f"- **Hardware profile:** `{result.get('hardware_profile', 'superconducting')}`\n"
        return output

    def run_quantum_simulation(self, circuit_json: str, n_qubits: int = 2) -> str:
        """
        Esegue un circuito quantistico personalizzato sul simulatore QuantumA Core.
        Usa questo tool quando l'utente chiede di simulare gate quantistici o algoritmi.

        Gate supportati (singolo qubit): H, X, Y, Z, S, T, RX, RY, RZ
        Gate supportati (due qubit): CNOT/CX, CZ, SWAP, iSWAP, RZZ, CRZ, CPHASE
        Gate supportati (tre qubit): CCNOT/TOFFOLI

        Per RX/RY/RZ/RZZ/CRZ/CPHASE usa il campo "params": {"theta": float} o {"phi": float}.

        Parametri attesi:
        - circuit_json: JSON string della lista gate. DEVE essere un array.
          Esempio: [{"gate": "H", "qubits": [0]}, {"gate": "CNOT", "qubits": [0, 1]}]
          Con parametri: [{"gate": "RY", "qubits": [0], "params": {"theta": 1.5708}}]
        - n_qubits: Numero totale di qubit (1-30; max 18 per density_matrix).
        """
        if not circuit_json or circuit_json.strip() == "":
            return (
                "❌ ERRORE: Hai inviato un circuit_json vuoto. "
                "Questo sarebbe un circuito banale (stato iniziale invariato), quindi non è un vero calcolo. "
                'Devi generare una stringa JSON valida (es. [{"gate": "H", "qubits": [0]}]).'
            )

        try:
            circuit = self._clean_json(circuit_json)
        except json.JSONDecodeError as e:
            return (
                "❌ ERRORE JSON: La stringa generata non è un JSON valido. "
                f"Dettaglio: {e}. Correggi la formattazione e riprova."
            )
        except ValueError as e:
            return f"❌ ERRORE JSON: {e}"

        if len(circuit) == 0:
            return (
                f"⚠️ CIRCUITO BANALE: hai inviato una lista vuota di istruzioni per {n_qubits} qubit. "
                "Il simulatore restituirebbe semplicemente |000...000⟩ senza applicare gate. "
                "Aggiungi almeno un gate, ad esempio H + CNOT, per ottenere un risultato utile."
            )

        trivial_circuit = all(
            isinstance(g, dict) and str(g.get("gate", "")).upper() == "MEASURE"
            for g in circuit
        )

        payload = {
            "n_qubits": n_qubits,
            "mode": "statevector",
            "backend": "auto",
            "instructions": circuit,
            "shots": 1024,
            "hardware_profile": "superconducting",
        }

        try:
            response = requests.post(
                f"{self.api_url}/simulate", json=payload, timeout=60
            )
            result = response.json()

            if response.status_code == 200 and result.get("success"):
                output = self._format_simulation_result(result, n_qubits)
                output += f"- **Circuito banale:** {'Sì' if trivial_circuit or result.get('gate_count', 0) == 0 else 'No'}\n"
                output += "\n_Note: se il circuito è banale o quasi banale, lo stato finale può restare concentrato su |000...000⟩._"
                return output

            return (
                f"❌ Errore dal simulatore: {json.dumps(result, indent=2, ensure_ascii=False)}\n\n"
                "⚠️ RICORDA: Usa gate validi ('H', 'X', 'CNOT', 'RY', 'RZ', ecc.) e metti gli angoli in `params`."
            )

        except requests.exceptions.RequestException as e:
            return (
                f"⚠️ Errore di connessione a QuantumA Core.\n"
                f"Target URL: {self.api_url}\n"
                "Assicurati che il server sia attivo.\n"
                f"Errore tecnico: {str(e)}"
            )

    def run_grover(
        self,
        n_qubits: int = 3,
        target_state: int = 5,
        iterations: int = 1,
        hardware_profile: str = "superconducting",
    ) -> str:
        """
        Esegue l'algoritmo di Grover pre-configurato su QuantumA Core.
        Usa questo tool quando l'utente vuole cercare uno stato specifico con l'algoritmo di Grover.

        Parametri:
        - n_qubits: numero di qubit (2-20).
        - target_state: indice intero dello stato target (0 <= target_state < 2^n_qubits).
        - iterations: numero di iterazioni di Grover (>=1). Ottimale: floor(pi/4 * sqrt(2^n_qubits)).
        - hardware_profile: superconducting | trapped_ion | silicon_spin | neutral_atom.
        """
        payload = {
            "n_qubits": n_qubits,
            "target_state": target_state,
            "iterations": iterations,
            "shots": 1024,
            "mode": "statevector",
            "hardware_profile": hardware_profile,
        }

        try:
            response = requests.post(
                f"{self.api_url}/grover", json=payload, timeout=60
            )
            result = response.json()

            if response.status_code == 200 and result.get("success"):
                output = f"### 🔍 Grover's Algorithm ({n_qubits} qubit, target=|{target_state:0{n_qubits}b}>)\n"
                output += f"- **Iterazioni:** {iterations}\n"
                output += self._format_simulation_result(result, n_qubits)
                return output

            return (
                f"❌ Errore da /grover: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )

        except requests.exceptions.RequestException as e:
            return (
                f"⚠️ Errore di connessione a QuantumA Core.\n"
                f"Target URL: {self.api_url}\n"
                f"Errore tecnico: {str(e)}"
            )

    def run_bell(
        self,
        n_qubits: int = 2,
        qubit_a: int = 0,
        qubit_b: int = 1,
        mode: str = "statevector",
        hardware_profile: str = "superconducting",
    ) -> str:
        """
        Prepara uno stato Bell (entanglement massimale) tra due qubit su QuantumA Core.
        Usa questo tool quando l'utente vuole generare o analizzare uno stato di Bell.

        Parametri:
        - n_qubits: numero totale di qubit (2-10).
        - qubit_a: primo qubit del pair (default 0).
        - qubit_b: secondo qubit del pair (default 1).
        - mode: statevector | density_matrix | monte_carlo.
        - hardware_profile: superconducting | trapped_ion | silicon_spin | neutral_atom.
        """
        payload = {
            "n_qubits": n_qubits,
            "qubit_a": qubit_a,
            "qubit_b": qubit_b,
            "shots": 1024,
            "mode": mode,
            "hardware_profile": hardware_profile,
        }

        try:
            response = requests.post(
                f"{self.api_url}/bell", json=payload, timeout=60
            )
            result = response.json()

            if response.status_code == 200 and result.get("success"):
                output = f"### 🔔 Bell State (qubit {qubit_a}–{qubit_b}, {n_qubits} qubit totali)\n"
                output += self._format_simulation_result(result, n_qubits)
                return output

            return (
                f"❌ Errore da /bell: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )

        except requests.exceptions.RequestException as e:
            return (
                f"⚠️ Errore di connessione a QuantumA Core.\n"
                f"Target URL: {self.api_url}\n"
                f"Errore tecnico: {str(e)}"
            )

    def suggest_quantum_template(
        self, template: str = "ghz", n_qubits: int = 27
    ) -> str:
        """
        Suggerisce un circuito o tool di partenza quando la richiesta è troppo generica.
        Template disponibili: ghz, bell, cat, simple_hadamard, grover_demo.
        """
        template = (template or "ghz").strip().lower()

        if template == "bell":
            circuit = '[{"gate":"H","qubits":[0]},{"gate":"CNOT","qubits":[0,1]}]'
            note = "Bell state a 2 qubit"
            hint = "Oppure usa il tool `run_bell` per una preparazione dedicata con noise model."
        elif template == "cat":
            circuit = '[{"gate":"H","qubits":[0]},{"gate":"CNOT","qubits":[0,1]},{"gate":"CNOT","qubits":[1,2]}]'
            note = "Cat state a 3 qubit"
            hint = ""
        elif template == "simple_hadamard":
            circuit = '[{"gate":"H","qubits":[0]}]'
            note = "Superposizione su un singolo qubit"
            hint = ""
        elif template == "grover_demo":
            circuit = None
            note = "Grover demo (3 qubit, target |101⟩)"
            hint = (
                "Usa il tool `run_grover` con: n_qubits=3, target_state=5, iterations=1.\n"
                "Non passare questo come circuit_json a run_quantum_simulation."
            )
        else:
            circuit = (
                '[{"gate":"H","qubits":[0]},'
                + ",".join(
                    f'{{"gate":"CNOT","qubits":[{i},{i+1}]}}'
                    for i in range(min(n_qubits - 1, 3))
                )
                + "]"
            )
            note = f"GHZ/cat state iniziale, adattabile fino a {n_qubits} qubit"
            hint = ""

        output = f"### 🧩 Template suggerito ({note})\n"
        output += f"- **n_qubits consigliati:** {n_qubits}\n"
        if circuit:
            output += f"- **circuit_json:** `{circuit}`\n"
        if hint:
            output += f"- **Nota:** {hint}\n"
        output += "- **Obiettivo:** partire da un circuito non banale e poi espanderlo.\n"
        return output

    def get_gpu_diagnostics(self) -> str:
        """
        Controlla lo stato del simulatore e della GPU NVIDIA.
        Utile per verificare se il sistema è pronto o per conoscere l'hardware attivo.
        """
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            data = response.json()
            gpu = data.get("gpu", {})

            output = "### 🖥️ Stato QuantumA Core\n"
            output += f"- **Status:** {str(data.get('status', 'unknown')).upper()}\n"
            output += f"- **GPU:** {gpu.get('device_name', 'N/D')}\n"
            output += f"- **CUDA:** {'Disponibile ✅' if gpu.get('cuda_available') else 'Non disponibile ❌'}\n"
            output += f"- **PyTorch:** {gpu.get('torch_version', 'N/D')}\n"
            output += f"- **Device count:** {gpu.get('device_count', 'N/D')}\n"
            output += f"- **Max qubit statevector:** {data.get('max_qubits_statevector', 'N/D')}\n"
            output += f"- **Max qubit density_matrix:** {data.get('max_qubits_density_matrix', 'N/D')}\n"
            output += f"- **Modi supportati:** {', '.join(data.get('supported_modes', []))}\n"
            output += f"- **Piattaforme supportate:** {', '.join(data.get('supported_platforms', []))}\n"

            if gpu.get("fallback_reason"):
                output += f"- **⚠️ Fallback CPU:** {gpu['fallback_reason']}\n"

            return output
        except Exception as e:
            return (
                f"❌ Il server QuantumA Core non risponde su `{self.api_url}`.\n"
                f"Assicurati che il server sia in esecuzione.\n"
                f"Errore: {str(e)}"
            )

    def get_platforms(self) -> str:
        """
        Restituisce i profili hardware disponibili (noise model) con i parametri fisici reali.
        Utile per scegliere quale hardware_profile usare nelle simulazioni.
        Piattaforme disponibili: superconducting, trapped_ion, silicon_spin, neutral_atom.
        """
        try:
            response = requests.get(f"{self.api_url}/platforms", timeout=5)
            platforms = response.json()

            output = "### ⚙️ Profili Hardware Disponibili\n"
            for p in platforms:
                output += f"\n**{p.get('name', '?')}**\n"
                output += f"- T1: {p.get('t1_us', 'N/D')} μs | T2: {p.get('t2_us', 'N/D')} μs\n"
                output += f"- Fidelity 1Q: {p.get('fidelity_single', 0):.4f} | 2Q: {p.get('fidelity_two', 0):.4f}\n"
                output += f"- Gate time 1Q: {p.get('gate_time_single_ns', 'N/D')} ns | 2Q: {p.get('gate_time_two_ns', 'N/D')} ns\n"
            return output

        except requests.exceptions.RequestException as e:
            return (
                f"⚠️ Errore di connessione a QuantumA Core.\n"
                f"Target URL: {self.api_url}\n"
                f"Errore tecnico: {str(e)}"
            )
