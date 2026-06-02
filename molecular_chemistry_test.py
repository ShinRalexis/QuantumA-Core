"""
Chimica quantistica reale: VQE per la molecola H2 su QuantumA Core.

Calcola l'energia dello stato fondamentale di H2 nella base STO-3G, interamente
da principi primi (nessuna libreria di chimica esterna, solo NumPy + il
simulatore QuantumA Core):

  1. Integrali molecolari STO-3G (overlap, cinetica, attrazione nucleare, ERI)
  2. Hartree-Fock ristretto (RHF) auto-consistente
  3. Hamiltoniano in seconda quantizzazione -> mapping Jordan-Wigner (4 qubit)
  4. VQE: ansatz a 1 parametro eseguito su QuantumA Core; per ogni angolo si
     estrae lo statevector dal simulatore e si calcola <H>; si minimizza.
  5. FCI esatto (diagonalizzazione) e curva di dissociazione come confronto.

Checkpoint di validazione (valori di letteratura, R=1.4 Bohr = 0.741 A):
  - overlap S01 ~ 0.659
  - E_HF  ~ -1.1167 Ha
  - E_FCI ~ -1.1373 Ha   (VQE deve raggiungerlo)
  - lunghezza di legame d'equilibrio ~ 0.741 A
"""

import sys
from pathlib import Path
from math import erf, pi, sqrt, exp

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_core.simulator import QuantumSimulator, SimulationMode
from quantum_core.circuit import QuantumCircuit

HARTREE_TO_KCAL = 627.509
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177

# ─── Base STO-3G per l'idrogeno (costanti standard EMSL) ──────────────────
_EXPS = np.array([3.42525091, 0.62391373, 0.16885540])
_COEF = np.array([0.15432897, 0.53532814, 0.44463454])
_DCOEF = _COEF * np.array([(2.0 * a / pi) ** 0.75 for a in _EXPS])  # normalizzati


# ─── Integrali su gaussiane s-type (formule di Szabo-Ostlund) ─────────────
def _boys0(t):
    """Funzione di Boys F0(t)."""
    if t < 1e-12:
        return 1.0
    return 0.5 * sqrt(pi / t) * erf(sqrt(t))


def _gprod(a, A, b, B):
    p = a + b
    P = (a * A + b * B) / p
    AB2 = float(np.dot(A - B, A - B))
    return p, P, AB2, exp(-a * b / p * AB2)


def _S(a, A, b, B):
    p, P, AB2, K = _gprod(a, A, b, B)
    return (pi / p) ** 1.5 * K


def _T(a, A, b, B):
    p, P, AB2, K = _gprod(a, A, b, B)
    return a * b / p * (3.0 - 2.0 * a * b / p * AB2) * (pi / p) ** 1.5 * K


def _V(a, A, b, B, C, Z):
    p, P, AB2, K = _gprod(a, A, b, B)
    return -2.0 * pi / p * Z * K * _boys0(p * float(np.dot(P - C, P - C)))


def _ERI(a, A, b, B, c, C, d, D):
    p, P, AB2, K1 = _gprod(a, A, b, B)
    q, Q, CD2, K2 = _gprod(c, C, d, D)
    return (2.0 * pi ** 2.5 / (p * q * sqrt(p + q)) * K1 * K2
            * _boys0(p * q / (p + q) * float(np.dot(P - Q, P - Q))))


def _contract2(func, cu, cv, *extra):
    tot = 0.0
    for i, a in enumerate(_EXPS):
        for j, b in enumerate(_EXPS):
            tot += _DCOEF[i] * _DCOEF[j] * func(a, cu, b, cv, *extra)
    return tot


def _contract_eri(cu, cv, cl, cs):
    tot = 0.0
    for i, a in enumerate(_EXPS):
        for j, b in enumerate(_EXPS):
            for k, c in enumerate(_EXPS):
                for m, d in enumerate(_EXPS):
                    tot += (_DCOEF[i] * _DCOEF[j] * _DCOEF[k] * _DCOEF[m]
                            * _ERI(a, cu, b, cv, c, cl, d, cs))
    return tot


def build_h2_integrals(R=1.4):
    """Integrali AO per H2 con i due nuclei a distanza R (Bohr)."""
    cen = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, R])]
    S = np.zeros((2, 2)); T = np.zeros((2, 2)); V = np.zeros((2, 2))
    for u in range(2):
        for v in range(2):
            S[u, v] = _contract2(_S, cen[u], cen[v])
            T[u, v] = _contract2(_T, cen[u], cen[v])
            V[u, v] = (_contract2(_V, cen[u], cen[v], cen[0], 1.0)
                       + _contract2(_V, cen[u], cen[v], cen[1], 1.0))
    Hcore = T + V
    ERI = np.zeros((2, 2, 2, 2))
    for u in range(2):
        for v in range(2):
            for l in range(2):
                for s in range(2):
                    ERI[u, v, l, s] = _contract_eri(cen[u], cen[v], cen[l], cen[s])
    return S, Hcore, ERI, 1.0 / R  # ultimo = repulsione nucleare


def rhf(S, Hcore, ERI, Enuc, nelec=2, maxit=200, tol=1e-11):
    """Hartree-Fock ristretto (closed shell)."""
    sval, svec = np.linalg.eigh(S)
    X = svec @ np.diag(sval ** -0.5) @ svec.T
    P = np.zeros_like(Hcore)
    Eold, nocc = 0.0, nelec // 2
    C = eps = None
    for _ in range(maxit):
        G = np.zeros_like(Hcore)
        for u in range(2):
            for v in range(2):
                G[u, v] = sum(P[l, s] * (ERI[u, v, s, l] - 0.5 * ERI[u, l, s, v])
                              for l in range(2) for s in range(2))
        F = Hcore + G
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        Pn = np.zeros_like(P)
        for u in range(2):
            for v in range(2):
                Pn[u, v] = 2.0 * sum(C[u, m] * C[v, m] for m in range(nocc))
        E_elec = 0.5 * np.sum(Pn * (Hcore + F))
        if abs(E_elec - Eold) < tol:
            P = Pn
            break
        P, Eold = Pn, E_elec
    return E_elec + Enuc, C, eps


# ─── Hamiltoniano qubit via Jordan-Wigner (4 spin-orbitali) ───────────────
_I2 = np.eye(2, dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # sigma- : |1>->|0>
_NSO = 4


def _kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _annih(p):
    """a_p con stringa di Jordan-Wigner (qubit 0 = MSB, come QuantumA Core)."""
    return _kron([_Z if q < p else (_SM if q == p else _I2) for q in range(_NSO)])


_A = [_annih(p) for p in range(_NSO)]
_AD = [op.conj().T for op in _A]


def build_qubit_hamiltonian(C, Hcore, ERI_AO, Enuc):
    """Hamiltoniano molecolare 16x16 nei 4 qubit (spin-orbitali) via JW."""
    h_MO = C.T @ Hcore @ C
    ERI_MO = np.einsum('up,vq,lr,ws,uvlw->pqrs', C, C, C, C, ERI_AO, optimize=True)
    dim = 1 << _NSO
    H = np.zeros((dim, dim), dtype=complex)
    for p in range(_NSO):
        for q in range(_NSO):
            if p % 2 == q % 2:
                H += h_MO[p // 2, q // 2] * (_AD[p] @ _A[q])
    for p in range(_NSO):
        for q in range(_NSO):
            for r in range(_NSO):
                for s in range(_NSO):
                    if p % 2 == q % 2 and r % 2 == s % 2:
                        v = ERI_MO[p // 2, q // 2, r // 2, s // 2]
                        if abs(v) > 1e-14:
                            H += 0.5 * v * (_AD[p] @ _AD[r] @ _A[s] @ _A[q])
    H = H + Enuc * np.eye(dim)
    return (H + H.conj().T) / 2  # hermitianizza (rumore numerico)


def fci_ground_energy(H):
    """Energia FCI: autovalore piu' basso nel settore a 2 elettroni."""
    dim = H.shape[0]
    w2 = [i for i in range(dim) if bin(i).count('1') == 2]
    return float(np.linalg.eigvalsh(H[np.ix_(w2, w2)].real)[0])


# ─── VQE eseguito su QuantumA Core ────────────────────────────────────────
def _ansatz_circuit(theta):
    """
    Ansatz UCCSD a 1 parametro per H2.

    Prepara cos(theta/2)|0011> + sin(theta/2)|1100>, ossia il sottospazio
    {HF, doppia eccitazione} che contiene lo stato fondamentale di H2.
    """
    qc = QuantumCircuit(4)
    qc.ry(0, theta=theta)
    qc.cx(0, 1)
    qc.x(0); qc.cx(0, 2); qc.cx(0, 3); qc.x(0)
    return qc


def vqe_energy(H, sim=None, n_scan=73):
    """
    Minimizza <H> sull'ansatz eseguito da QuantumA Core.

    Ritorna (energia_minima, theta_ottimo). L'ottimizzazione e' una scansione
    grossolana + raffinamento (un solo parametro, superficie liscia).
    """
    if sim is None:
        sim = QuantumSimulator(4, SimulationMode.STATEVECTOR, backend="auto", seed=1)

    def E(theta):
        sim.run_circuit(_ansatz_circuit(theta), shots=1)
        eng = sim.engine
        psi = eng.to_numpy() if hasattr(eng, 'to_numpy') else np.asarray(eng.amplitudes)
        return float(np.real(np.vdot(psi, H @ psi)))

    thetas = np.linspace(0, 2 * pi, n_scan)
    energies = [E(t) for t in thetas]
    i0 = int(np.argmin(energies))
    lo = thetas[max(0, i0 - 1)]; hi = thetas[min(n_scan - 1, i0 + 1)]
    fine = np.linspace(lo, hi, 101)
    fe = [E(t) for t in fine]
    j0 = int(np.argmin(fe))
    return fe[j0], fine[j0]


# ─── Funzioni di alto livello ─────────────────────────────────────────────
def run_equilibrium(R=1.4, verbose=True):
    """Calcolo completo all'equilibrio: HF, FCI, VQE su QuantumA Core."""
    S, Hcore, ERI, Enuc = build_h2_integrals(R)
    E_HF, C, eps = rhf(S, Hcore, ERI, Enuc)
    H = build_qubit_hamiltonian(C, Hcore, ERI, Enuc)
    E_FCI = fci_ground_energy(H)
    E_VQE, theta_opt = vqe_energy(H)
    E_corr = E_FCI - E_HF

    if verbose:
        print("=" * 64)
        print("   H2 / STO-3G  -  VQE su QuantumA Core")
        print("=" * 64)
        print(f"Geometria       : R = {R} Bohr = {R * BOHR_TO_ANGSTROM:.4f} Angstrom")
        print(f"Overlap S[0,1]  : {S[0, 1]:.6f}   (atteso ~0.659)")
        print(f"E_nuc           : {Enuc:.6f} Ha")
        print("-" * 64)
        print(f"Hartree-Fock    : {E_HF:.6f} Ha   (atteso ~ -1.1167)")
        print(f"VQE (QuantumA)  : {E_VQE:.6f} Ha   (theta = {theta_opt:.4f})")
        print(f"FCI (esatto)    : {E_FCI:.6f} Ha   (atteso ~ -1.1373)")
        print(f"Sperimentale    : ~ -1.1744 Ha")
        print("-" * 64)
        print(f"Energia di correlazione (FCI-HF): {E_corr * 1000:.3f} mHa "
              f"= {E_corr * HARTREE_TO_KCAL:.2f} kcal/mol")
        print(f"Errore VQE vs FCI: {abs(E_VQE - E_FCI) * 1e6:.3f} microHartree")
        ok = abs(E_FCI - (-1.1373)) < 3e-3 and abs(E_VQE - E_FCI) < 1e-4
        print(f"CHECKPOINT: {'OK - VQE riproduce il valore esatto' if ok else 'FALLITO'}")
        print("=" * 64)

    return {"E_HF": E_HF, "E_VQE": E_VQE, "E_FCI": E_FCI, "E_corr": E_corr}


def dissociation_curve(Rs=(0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0)):
    """Superficie di energia potenziale di H2 (HF vs FCI) lungo il legame."""
    rows = []
    for R in Rs:
        S, Hcore, ERI, Enuc = build_h2_integrals(R)
        E_HF, C, eps = rhf(S, Hcore, ERI, Enuc)
        E_FCI = fci_ground_energy(build_qubit_hamiltonian(C, Hcore, ERI, Enuc))
        rows.append((R, E_HF, E_FCI))

    print("\n" + "=" * 60)
    print("   CURVA DI DISSOCIAZIONE H2 (HF vs FCI)")
    print("=" * 60)
    print(f"{'R[Bohr]':>8} {'R[A]':>7} {'E_HF':>11} {'E_FCI':>11} {'corr[mHa]':>10}")
    for R, hf, fci in rows:
        print(f"{R:8.2f} {R * BOHR_TO_ANGSTROM:7.3f} {hf:11.6f} {fci:11.6f} "
              f"{(fci - hf) * 1000:10.2f}")
    fci_vals = [r[2] for r in rows]
    imin = int(np.argmin(fci_vals))
    Re, Emin = rows[imin][0], fci_vals[imin]
    Ediss = rows[-1][2]
    print("-" * 60)
    print(f"Equilibrio: R_e ~ {Re * BOHR_TO_ANGSTROM:.3f} A, E = {Emin:.5f} Ha")
    print(f"Energia di legame D_e ~ {(Ediss - Emin) * HARTREE_TO_KCAL:.1f} kcal/mol")
    print("=" * 60)
    return rows


if __name__ == "__main__":
    run_equilibrium()
    dissociation_curve()
