# HF-ENERGY-OF-H2
import numpy as np

# Verilen veriler
S = np.array([[1.0000000, 0.6589309],
              [0.6589309, 1.0000000]])

h = np.array([[-1.1200252, -0.9576853],
              [-0.9576853, -1.1200252]])

two_electron_integrals = {
    (0, 0, 0, 0): 0.7746059,
    (1, 1, 1, 1): 0.7746059,
    (1, 0, 0, 0): 0.4437704,
    (1, 1, 0, 1): 0.4437704,
    (1, 0, 1, 0): 0.2966367,
    (1, 1, 0, 0): 0.5694534
}

nuclear_repulsion_energy = 0.7137155
convergence_tolerance = 1e-4

# İteratif SCF Prosedürü
def scf_procedure(S, h, two_electron_integrals, nuclear_repulsion_energy, convergence_tolerance):
    # Overlap matrix S'nin eigenvalue decomposition'u
    eigvals, eigvecs = np.linalg.eigh(S)
    S_half_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Başlangıç yoğunluk matrisi sıfır
    P = np.zeros_like(h)

    def fock_matrix(P):
        F = h.copy()
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        F[p, q] += P[r, s] * (
                            two_electron_integrals.get((p, q, r, s), 0) -
                            0.5 * two_electron_integrals.get((p, s, r, q), 0)
                        )
        return F

    iteration = 0
    while True:
        F = fock_matrix(P)
        F_prime = S_half_inv @ F @ S_half_inv
        eigvals, C_prime = np.linalg.eigh(F_prime)
        C = S_half_inv @ C_prime
        P_new = np.zeros_like(P)
        for i in range(2):
            for j in range(2):
                P_new[i, j] = 2 * sum(C[i, k] * C[j, k] for k in range(1))

        if np.linalg.norm(P_new - P) < convergence_tolerance:
            break
        P = P_new
        iteration += 1

    energy = 0.5 * np.sum(P * (h + F)) + nuclear_repulsion_energy
    return energy, iteration

energy, iteration = scf_procedure(S, h, two_electron_integrals, nuclear_repulsion_energy, convergence_tolerance)
energy, iteration
