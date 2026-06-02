"""
Test del simulatore quantistico -- Verifica funzionalita core.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit
import numpy as np


def test_bell_state():
    """Test stato Bell (|00> + |11>)/sqrt(2)."""
    print("\n[TEST] Bell State")
    
    sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
    result = sim.run_bell_state(shots=4096)
    
    top_states = result.most_likely_states(5)
    print(f"   Top states: {top_states[:3]}")
    
    assert len(top_states) >= 2, "Bell state deve produrre almeno 2 stati"
    
    state_keys = [s[0] for s in top_states]
    assert '00' in state_keys, f"|00> mancante. States: {state_keys}"
    assert '11' in state_keys, f"|11> mancante. States: {state_keys}"
    
    for state, prob in top_states[:2]:
        print(f"   -> {state}: {prob:.4f}")
    
    assert abs(result.circuit_fidelity - 1.0) < 0.01, \
        f"Fidelity troppo bassa: {result.circuit_fidelity}"
    
    print("   [OK] PASS")


def test_ghz_state():
    """Test stato GHZ (|00...0> + |11...1>)/sqrt(2)."""
    print("\n[TEST] GHZ State")
    
    sim = QuantumSimulator(4, SimulationMode.STATEVECTOR)
    result = sim.run_ghz_state(shots=4096)
    
    top_states = result.most_likely_states(5)
    state_keys = [s[0] for s in top_states]
    print(f"   Top states: {top_states[:3]}")
    
    assert '0000' in state_keys, f"|0000> mancante. States: {state_keys}"
    assert '1111' in state_keys, f"|1111> mancante. States: {state_keys}"
    
    print("   [OK] PASS")


def test_custom_circuit():
    """Test circuito custom."""
    print("\n[TEST] Custom Circuit (W State)")
    
    circuit = QuantumCircuit(3, name="w_state")
    circuit.ry(0, 2 * np.arcsin(1 / np.sqrt(3)))
    circuit.cx(0, 1)
    circuit.rx(1, -2 * np.arcsin(1 / np.sqrt(2)))
    
    sim = QuantumSimulator(3, SimulationMode.STATEVECTOR)
    result = sim.run_circuit(circuit, shots=4096)
    
    top_states = result.most_likely_states(5)
    print(f"   Top states: {top_states[:5]}")
    
    assert len(top_states) >= 2, f"Troppi pochi stati. Got: {len(top_states)}"
    
    for state, prob in top_states[:3]:
        print(f"   -> {state}: {prob:.4f}")
    
    print("   [OK] PASS")


def test_density_matrix_noise():
    """Test density matrix con rumore T1/T2."""
    print("\n[TEST] Density Matrix (Noise)")
    
    sim = QuantumSimulator(3, SimulationMode.DENSITY_MATRIX, "superconducting")
    
    circuit = QuantumCircuit(3, name="noisy_bell")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.s(2)
    circuit.t(2)
    
    result = sim.run_circuit(circuit, shots=1024)
    
    print(f"   Fidelity: {result.circuit_fidelity:.6f}")
    print(f"   Error rate: {result.estimated_error_rate:.4%}")
    print(f"   Top states: {result.most_likely_states(3)}")
    
    assert result.circuit_fidelity <= 1.0, "Fidelity non dovrebbe superare 1.0"
    
    stats = sim.get_stats()
    print(f"   Stats: {stats}")
    
    print("   [OK] PASS")


def test_grover():
    """Test algoritmo di Grover."""
    print("\n[TEST] Grover Algorithm")
    
    sim = QuantumSimulator(3, SimulationMode.STATEVECTOR)
    result = sim.run_grover(n_iterations=1, target_state=5, shots=4096)
    
    top_states = result.most_likely_states(5)
    print(f"   Top states: {top_states[:5]}")
    
    for state, prob in top_states[:3]:
        print(f"   -> {state}: {prob:.4f}")
    
    assert result.circuit_fidelity > 0.9, \
        f"Fidelity troppo bassa per Grover: {result.circuit_fidelity}"
    
    print("   [OK] PASS")


def test_optimization():
    """Test gate fusion optimization."""
    print("\n[TEST] Circuit Optimization (Gate Fusion)")
    
    circuit = QuantumCircuit(2, name="optimize_test")
    circuit.h(0)
    circuit.h(0)  # Should cancel with previous H
    circuit.x(1)
    circuit.x(1)  # Should cancel with previous X
    circuit.h(0)
    circuit.s(0)
    
    optimized = circuit.optimize()
    
    print(f"   Original gates: {circuit.total_gates}")
    print(f"   Optimized gates: {optimized.total_gates}")
    
    assert optimized.total_gates <= circuit.total_gates, \
        "Il circuito ottimizzato dovrebbe avere <= gate"
    
    print(f"   Optimized circuit gates: {optimized.gate_count}")
    print("   [OK] PASS")


def test_measurement():
    """Test misurazione e probability."""
    print("\n[TEST] Measurement & Probabilities")
    
    sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
    
    circuit = QuantumCircuit(2, name="measure_test")
    circuit.x(1)  # |01>
    
    result = sim.run_circuit(circuit, shots=4096)
    
    top_states = result.most_likely_states(5)
    print(f"   Top states: {top_states[:3]}")
    
    assert '01' in [s[0] for s in top_states], \
        f"|01> mancante. States: {[s[0] for s in top_states]}"
    
    prob_01 = result.probability_distribution.get('01', 0)
    print(f"   P(|01>): {prob_01:.4f}")
    assert prob_01 > 0.95, f"P(|01>) troppo bassa: {prob_01}"
    
    print("   [OK] PASS")


def test_bloch_sphere():
    """Test coordinate di Bloch."""
    print("\n[TEST] Bloch Sphere Coordinates")
    
    sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
    
    circuit = QuantumCircuit(2, name="bloch_test")
    circuit.h(0)
    
    result = sim.run_circuit(circuit, shots=1024)
    
    bloch = sim.bloch_sphere(0)
    print(f"   Bloch vector for H(q0): X={bloch[0]:.3f}, "
          f"Y={bloch[1]:.3f}, Z={bloch[2]:.3f}")
    
    assert abs(bloch[2]) < 0.1, \
        f"<Z> dopo H dovrebbe essere ~0, got {bloch[2]}"
    
    print("   [OK] PASS")


def main():
    """Esegue tutti i test."""
    
    print("=" * 50)
    print("QuantumA Core - Test Suite")
    print("=" * 50)
    
    tests = [
        test_bell_state,
        test_ghz_state,
        test_custom_circuit,
        test_density_matrix_noise,
        test_grover,
        test_optimization,
        test_measurement,
        test_bloch_sphere,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"   [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"   [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} test(s) failed -- see details above")


if __name__ == "__main__":
    main()
