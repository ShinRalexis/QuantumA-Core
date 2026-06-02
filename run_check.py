import sys
sys.path.insert(0, '.')
import numpy as np
import time

PASS = 0
FAIL = 0

def ok(name):
    global PASS; PASS += 1; print(f'  [OK]  {name}')

def err(name, e):
    global FAIL; FAIL += 1; print(f'  [FAIL] {name}: {e}')

print('=' * 60)
print('  QUANTUMA CORE -- FULL FUNCTIONAL CHECK')
print('=' * 60)

# ── 1. IMPORTS ──────────────────────────────────────────────
print('\n[1] Module imports')
try:
    from quantum_core.statevector import StatevectorEngine; ok('StatevectorEngine')
except Exception as e: err('StatevectorEngine', e)
try:
    from quantum_core.density_matrix import DensityMatrixEngine; ok('DensityMatrixEngine')
except Exception as e: err('DensityMatrixEngine', e)
try:
    from quantum_core.gates import QuantumGateLibrary; ok('QuantumGateLibrary')
except Exception as e: err('QuantumGateLibrary', e)
try:
    from quantum_core.noise_models import NoiseModelLibrary, QubitParameters; ok('NoiseModelLibrary')
except Exception as e: err('NoiseModelLibrary', e)
try:
    from quantum_core.circuit import QuantumCircuit; ok('QuantumCircuit')
except Exception as e: err('QuantumCircuit', e)
try:
    from quantum_core.simulator import QuantumSimulator, SimulationMode; ok('QuantumSimulator')
except Exception as e: err('QuantumSimulator', e)
try:
    from quantum_core.gpu_statevector import GPUStatevectorEngine; ok('GPUStatevectorEngine (import)')
except Exception as e: err('GPUStatevectorEngine', e)

from quantum_core.statevector import StatevectorEngine
from quantum_core.density_matrix import DensityMatrixEngine
from quantum_core.gates import QuantumGateLibrary
from quantum_core.noise_models import NoiseModelLibrary
from quantum_core.circuit import QuantumCircuit
from quantum_core.simulator import QuantumSimulator, SimulationMode

gl = QuantumGateLibrary()

# ── 2. GATE LIBRARY ─────────────────────────────────────────
print('\n[2] Gate library matrices')
try:
    for g in ['H', 'X', 'Y', 'Z', 'S', 'T']:
        m = gl.get_matrix(g)
        assert np.allclose(m @ m.conj().T, np.eye(2), atol=1e-12), f'{g} not unitary'
    ok('Single-qubit gates unitary (H,X,Y,Z,S,T)')
except Exception as e: err('Single-qubit unitarity', e)

try:
    m = gl.get_matrix('RY', theta=np.pi)
    v = m @ np.array([1, 0])
    assert abs(abs(v[1]) - 1.0) < 1e-10, f'RY(pi)|0> wrong: {v}'
    ok('RY(pi) correct direction')
except Exception as e: err('RY matrix', e)

try:
    m = gl.rzz_matrix(np.pi / 2)
    assert np.allclose(m @ m.conj().T, np.eye(4), atol=1e-12)
    ok('RZZ unitary')
except Exception as e: err('RZZ matrix', e)

try:
    for g in ['CNOT', 'CZ', 'SWAP', 'iSWAP', 'CRZ']:
        m = gl.get_two_qubit_gate(g) if g != 'CRZ' else gl.crz_matrix(np.pi / 4)
        assert np.allclose(m @ m.conj().T, np.eye(4), atol=1e-12)
    ok('Two-qubit gates unitary (CNOT,CZ,SWAP,iSWAP,CRZ)')
except Exception as e: err('Two-qubit unitarity', e)

try:
    m = gl.ccnot_matrix()
    assert np.allclose(m @ m.conj().T, np.eye(8), atol=1e-12)
    ok('CCNOT (Toffoli) unitary 8x8')
except Exception as e: err('CCNOT matrix', e)

# ── 3. STATEVECTOR ENGINE ───────────────────────────────────
print('\n[3] StatevectorEngine (CPU vectorized)')
try:
    sv = StatevectorEngine(2)
    sv.apply_gate('H', 0)
    sv.apply_two_qubit_gate('CNOT', 0, 1)
    p = sv.get_all_probabilities()
    assert abs(p[0] - 0.5) < 1e-10 and abs(p[3] - 0.5) < 1e-10
    ok('Bell state |00>+|11> correct probabilities')
except Exception as e: err('Bell state statevector', e)

try:
    sv2 = StatevectorEngine(3)
    sv2.apply_gate('H', 0)
    sv2.apply_gate('X', 2)
    sv2.apply_custom_gate(gl.ccnot_matrix(), [0, 1, 2])
    ok('CCNOT apply_custom_gate 3-qubit no crash')
except Exception as e: err('CCNOT apply_custom_gate', e)

try:
    sv3 = StatevectorEngine(1)
    sv3.apply_gate('H', 0)
    bx, by, bz = sv3.get_bloch_vector(0)
    assert abs(bx - 1.0) < 0.01 and abs(by) < 0.01 and abs(bz) < 0.01
    ok('Bloch vector H|0>: X=1 Y=0 Z=0')
except Exception as e: err('Bloch vector', e)

try:
    sv4 = StatevectorEngine(2)
    sv4.apply_two_qubit_gate('RZZ', 0, 1, {'theta': np.pi / 2})
    assert sv4.normalized
    ok('RZZ gate preserves normalization')
except Exception as e: err('RZZ gate', e)

try:
    sv5 = StatevectorEngine(2)
    sv5.apply_two_qubit_gate('CRZ', 0, 1, {'phi': np.pi})
    assert sv5.normalized
    ok('CRZ gate no crash')
except Exception as e: err('CRZ gate', e)

# ── 4. DENSITY MATRIX ────────────────────────────────────────
print('\n[4] DensityMatrixEngine')
try:
    dm = DensityMatrixEngine(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    dm.apply_unitary(H, [0])
    dm.apply_unitary(gl.cnot_matrix(), [0, 1])
    p = dm.get_probabilities()
    assert abs(p[0] - 0.5) < 0.01 and abs(p[3] - 0.5) < 0.01
    ok('Bell state density matrix')
except Exception as e: err('Bell density matrix', e)

try:
    dm2 = DensityMatrixEngine(1)
    dm2.apply_unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), [0])
    kraus = NoiseModelLibrary.amplitude_damping_kraus(0.1)
    dm2.apply_kraus_channel(kraus, [0])
    assert dm2.is_valid_density_matrix
    ok('Amplitude damping: valid density matrix after noise')
except Exception as e: err('Kraus channel', e)

try:
    dm3 = DensityMatrixEngine(2)
    x, y, z = dm3.bloch_coordinates(0)
    assert abs(z - 1.0) < 0.01
    ok('Bloch coordinates |0>: Z=1')
except Exception as e: err('Bloch coordinates DM', e)

try:
    dm4 = DensityMatrixEngine(3)
    dm4.apply_unitary(gl.ccnot_matrix(), [0, 1, 2])
    assert dm4.is_valid_density_matrix
    ok('CCNOT density matrix 3-qubit')
except Exception as e: err('CCNOT density matrix', e)

# ── 5. CIRCUIT BUILDER ────────────────────────────────────────
print('\n[5] QuantumCircuit builder')
try:
    qc = QuantumCircuit(3)
    (qc.h(0).cx(0, 1).cz(1, 2)
       .rx(0, theta=np.pi / 4).ry(1, theta=np.pi / 3).rz(2, phi=np.pi / 6)
       .swap(0, 1).cphase(0, 2, theta=np.pi / 8).rzz(0, 1, theta=np.pi / 4)
       .ccnot(0, 1, 2).measure([0, 1, 2]))
    assert qc.total_gates == 13
    ok(f'All gate types in circuit builder (13 gates)')
except Exception as e: err('Circuit builder', e)

try:
    qc2 = QuantumCircuit(2)
    for _ in range(4):
        qc2.h(0)
    qc2.s(0); qc2.s(0)
    opt = qc2.optimize()
    assert opt.total_gates < qc2.total_gates
    ok(f'Gate fusion/cancellation: {qc2.total_gates} -> {opt.total_gates} gates')
except Exception as e: err('Gate fusion', e)

try:
    ascii_out = qc.to_ascii()
    assert 'q0' in ascii_out and 'q2' in ascii_out
    ok('ASCII circuit rendering')
except Exception as e: err('ASCII render', e)

try:
    d = qc.to_dict()
    qc_back = QuantumCircuit.from_dict(d)
    assert qc_back.n_qubits == 3 and qc_back.total_gates == 13
    ok('Circuit to_dict / from_dict round-trip')
except Exception as e: err('Circuit serialization', e)

# ── 6. NOISE MODELS ──────────────────────────────────────────
print('\n[6] Noise models')
try:
    for name in ['superconducting', 'trapped_ion', 'silicon_spin', 'neutral_atom']:
        p = NoiseModelLibrary.get_platform(name)
        assert p.t1 > 0 and p.fidelity_single > 0
    ok('All 4 hardware profiles load correctly')
except Exception as e: err('Hardware profiles', e)

try:
    kraus = NoiseModelLibrary.combined_t1t2_kraus(0.03, 80.0, 60.0)
    s = sum(k.conj().T @ k for k in kraus)
    assert np.allclose(s, np.eye(2), atol=1e-6)
    ok('T1/T2 Kraus: sum(K-dagger K) = I')
except Exception as e: err('combined_t1t2_kraus normalization', e)

try:
    qc_err = QuantumCircuit(2)
    qc_err.h(0); qc_err.cx(0, 1)
    nm = NoiseModelLibrary()
    from quantum_core.noise_models import QubitParameters
    params = {0: QubitParameters(), 1: QubitParameters()}
    info = nm.calculate_circuit_error(qc_err.instructions, params)
    assert info['num_gates'] == 2
    ok(f'calculate_circuit_error: {info["num_gates"]} gates, fidelity={info["estimated_circuit_fidelity"]:.4f}')
except Exception as e: err('calculate_circuit_error', e)

# ── 7. SIMULATOR MODES ───────────────────────────────────────
print('\n[7] QuantumSimulator')
try:
    sim = QuantumSimulator(2, SimulationMode.STATEVECTOR)
    r = sim.run_bell_state()
    p = r.probability_distribution
    assert abs(p.get('00', 0) - 0.5) < 0.05 and abs(p.get('11', 0) - 0.5) < 0.05
    ok('STATEVECTOR Bell state: |00>=0.5 |11>=0.5')
except Exception as e: err('Statevector mode', e)

try:
    sim2 = QuantumSimulator(2, SimulationMode.DENSITY_MATRIX)
    qc_b = QuantumCircuit(2); qc_b.h(0); qc_b.cx(0, 1)
    r2 = sim2.run_circuit(qc_b, shots=256)
    ok(f'DENSITY_MATRIX mode: {r2.shots_completed} shots, fidelity={r2.circuit_fidelity:.4f}')
except Exception as e: err('Density matrix mode', e)

try:
    sim3 = QuantumSimulator(2, SimulationMode.MONTE_CARLO)
    qc_b2 = QuantumCircuit(2); qc_b2.h(0); qc_b2.cx(0, 1)
    r3 = sim3.run_circuit(qc_b2, shots=64)
    ok(f'MONTE_CARLO mode: {r3.shots_completed} shots')
except Exception as e: err('Monte Carlo mode', e)

try:
    sim4 = QuantumSimulator(4, SimulationMode.STATEVECTOR)
    r4 = sim4.run_grover(n_iterations=1, target_state=15)
    top = r4.most_likely_states(1)[0]
    ok(f'Grover 4-qubit target |1111>: top={top[0]} p={top[1]:.3f}')
except Exception as e: err('Grover algorithm', e)

try:
    sim5 = QuantumSimulator(3, SimulationMode.STATEVECTOR)
    qc_x = QuantumCircuit(3); qc_x.x(0)
    r_a = sim5.run_circuit(qc_x, shots=100)
    r_b = sim5.run_circuit(qc_x, shots=100)
    assert r_a.probability_distribution == r_b.probability_distribution
    ok('Engine reset between consecutive run_circuit calls')
except Exception as e: err('Engine reset', e)

try:
    sim6 = QuantumSimulator(4, SimulationMode.STATEVECTOR)
    qc6 = QuantumCircuit(4); qc6.h(0); qc6.cx(0, 1); qc6.cx(0, 2); qc6.cx(0, 3)
    sim6.run_circuit(qc6)
    ee = sim6.entanglement_entropy([0, 1])
    assert 0.0 <= ee <= 4.0
    ok(f'Entanglement entropy GHZ partition [0,1]: {ee:.4f} bits')
except Exception as e: err('Entanglement entropy', e)

try:
    sim7 = QuantumSimulator(3, SimulationMode.STATEVECTOR)
    qc7 = QuantumCircuit(3); qc7.h(0); qc7.cx(0, 1); qc7.cx(0, 2)
    sim7.run_circuit(qc7)
    bx, by, bz = sim7.bloch_sphere(0)
    ok(f'Bloch sphere GHZ q0: X={bx:.3f} Y={by:.3f} Z={bz:.3f}')
except Exception as e: err('Bloch sphere simulator', e)

# ── SUMMARY ─────────────────────────────────────────────────
print()
print('=' * 60)
print(f'  CHECK COMPLETE:  {PASS} passed  |  {FAIL} failed')
print('=' * 60)
