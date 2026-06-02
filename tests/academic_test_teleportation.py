
import numpy as np
from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

def academic_test_teleportation():
    """
    Academic Problem: Quantum Teleportation
    Goal: Teleport an arbitrary state from Qubit 0 to Qubit 2 using an entangled pair on (1, 2).
    """
    print("=== Academic Test: Quantum Teleportation ===")
    n_qubits = 3
    sim = QuantumSimulator(n_qubits, mode=SimulationMode.STATEVECTOR, backend='cuda')
    
    # 1. Prepare an arbitrary state on Qubit 0 to teleport
    # Let's teleport a state created by RY(pi/3) -> cos(pi/6)|0> + sin(pi/6)|1>
    theta = np.pi / 3
    qc = QuantumCircuit(n_qubits)
    qc.ry(0, theta) 
    
    # 2. Create entanglement between Qubit 1 and Qubit 2 (Alice and Bob)
    qc.h(1)
    qc.cx(1, 2)
    
    # 3. Alice performs operations on Qubit 0 and her half of the Bell pair (Qubit 1)
    qc.cx(0, 1)
    qc.h(0)
    
    # 4. Bob applies corrections based on Alice's results (represented as controlled gates)
    # Classically, Alice measures and tells Bob. In a circuit, this is CX and CZ.
    qc.cx(1, 2) # If Q1 is 1, apply X
    qc.cz(0, 2) # If Q0 is 1, apply Z
    
    print(f"Running teleportation circuit on {sim.backend} backend...")
    result = sim.run_circuit(qc, shots=4096)
    
    # 5. Verification
    # If teleportation worked, Qubit 2 should now have the Bloch vector that Qubit 0 had.
    # Initial Q0 state: RY(theta)|0>
    expected_x = np.sin(theta)
    expected_y = 0.0
    expected_z = np.cos(theta)
    
    # Get Bob's qubit (Q2) Bloch vector
    # Note: bloch_sphere computes the reduced state vector/density matrix for the qubit
    bloch = sim.bloch_sphere(2)
    
    print("\nResults:")
    print(f"Target Qubit (Q2) Bloch Vector: X={bloch[0]:.4f}, Y={bloch[1]:.4f}, Z={bloch[2]:.4f}")
    print(f"Expected Bloch Vector:       X={expected_x:.4f}, Y={expected_y:.4f}, Z={expected_z:.4f}")
    
    fidelity = result.circuit_fidelity
    print(f"Simulation Fidelity: {fidelity:.4f}")
    
    # Precision check
    diff = np.sqrt((bloch[0]-expected_x)**2 + (bloch[1]-expected_y)**2 + (bloch[2]-expected_z)**2)
    if diff < 1e-2:
        print("\n[SUCCESS] Quantum Teleportation verified! The state was successfully transferred to Bob.")
    else:
        print(f"\n[FAILURE] Teleportation precision error: {diff:.4f}")

def debug_gpu_bloch():
    print("=== Debug: GPU Bloch Vector H Gate ===")
    sim = QuantumSimulator(1, mode=SimulationMode.STATEVECTOR, backend='cuda')
    qc = QuantumCircuit(1)
    qc.h(0)
    sim.run_circuit(qc)
    bloch = sim.bloch_sphere(0)
    print(f"H(0) Bloch Vector: X={bloch[0]:.4f}, Y={bloch[1]:.4f}, Z={bloch[2]:.4f}")
    # Expected: X=1, Y=0, Z=0

def debug_gpu_ry():
    print("=== Debug: GPU Bloch Vector RY Gate ===")
    sim = QuantumSimulator(1, mode=SimulationMode.STATEVECTOR, backend='cuda')
    qc = QuantumCircuit(1)
    theta = np.pi / 3
    qc.ry(0, theta)
    sim.run_circuit(qc)
    bloch = sim.bloch_sphere(0)
    print(f"RY(pi/3) Bloch Vector: X={bloch[0]:.4f}, Y={bloch[1]:.4f}, Z={bloch[2]:.4f}")
    # Expected: X=-0.8660, Z=0.5000 (based on gates.py definition)

if __name__ == "__main__":
    debug_gpu_bloch()
    debug_gpu_ry()
    academic_test_teleportation()
