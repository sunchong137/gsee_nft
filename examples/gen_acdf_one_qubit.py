"""
Generate the approximate CDF for a one qubit Hamiltonian.
"""
from gsee import acdf
import numpy as np
from matplotlib import pyplot as plt

# Set the parameters
n_samp = 10000
max_dft_order = 40
rescaled_energy_acc = 0.01
nmesh = 10000
max_x = np.pi / 2
energy_grid = np.linspace(-max_x, max_x, nmesh, endpoint=True)
init_guess_noise = 1.5


# Set the Hamiltonian
# Generate a random real one-qubit Hamiltonian
# hamiltonian = np.random.rand(2, 2)
hamiltonian = np.array([[1, 0], [0, -0.5]])
hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)

# Solve the Hamiltonian
ew, ev = np.linalg.eigh(hamiltonian)
energy_rescalor = np.pi / 3
print("eigenvalues: E0 = {}; E1 = {}".format(ew[0], ew[1]))

# generate the initial guess based on the ground state ev[:0]
# state = ev[:, 0] + np.random.rand(2) * init_guess_noise
state = np.array([0.5, 0.5])
state /= np.linalg.norm(state)
p0 = np.dot(ev[:, 0].T, state) ** 2
print("The overlap between the initial state and the ground state: {}".format(p0))

# Perform the classical and quantum sampling separately
Fj, j_samp = acdf.classical_sampler(n_samp, max_dft_order, rescaled_energy_acc, nmesh=nmesh)
Z_samp = acdf.quantum_sampler(j_samp, state, hamiltonian, energy_rescalor=None)

# evaluate the ACDF
G_bar = acdf.acdf_kernel(
    max_dft_order, Fj, j_samp, Z_samp, energy_grid=energy_grid, nmesh=nmesh
)

# Plot the ACDF
plt.plot(energy_grid, G_bar)
plt.xticks(ticks=ew * energy_rescalor , labels=["E0", "E1"])
plt.xlabel(r"$energy$")
plt.ylabel(r"$ACDF$")
# plt.savefig("figures/G_bar.png", dpi=300)

# Plot the Fj
# plt.plot(
#     np.arange(-max_dft_order, max_dft_order + 1),
#     np.imag(Fj),
#     linestyle="",
#     marker="o",
#     markersize=3,
# )
# plt.plot(
#     np.arange(-max_dft_order, max_dft_order + 1),
#     np.real(Fj),
#     linestyle="",
#     marker="o",
#     markersize=3,
# )
# plt.xlabel(r"$index$")
# plt.ylabel(r"$|F_j|$")
# plt.yscale("log")

plt.show()
