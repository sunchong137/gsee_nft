"""
Generate the approximate CDF for a one qubit Hamiltonian.
"""
from gsee.gaussian_derivative import *
from gsee.acdf import *
import numpy as np
from matplotlib import pyplot as plt

# Set the parameters
n_samp_acdf = 4000
n_samp_gauss = 40000
acdf_max_dft_order = 1600
gaussian_derivative_max_dft_order = 80
rescaled_energy_acc = 0.02
nmesh = 1000
max_x = np.pi / 2
energy_grid = np.linspace(-max_x, max_x, nmesh)
# init_guess_noise = 1.5


# Set the Hamiltonian
# Generate a random real one-qubit Hamiltonian
# hamiltonian = np.random.rand(2, 2)
hamiltonian = np.array([[0.5, 0], [0, -0.5]])
hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)

# Solve the Hamiltonian
ew, ev = np.linalg.eigh(hamiltonian)
print("eigenvalues: E0 = {}; E1 = {}".format(ew[0], ew[1]))
energy_gap = np.abs(ew[0] - ew[1])

# generate the initial guess based on the ground state ev[:0]
# state = ev[:, 0] + np.random.rand(2) * init_guess_noise
state = np.array([4 / 5, 3 / 5])
state /= np.linalg.norm(state)
p0 = np.dot(ev[:, 0].T, state) ** 2
print("The overlap between the initial state and the ground state: {}".format(p0))


# ACDF algorithm

# Perform the classical and quantum sampling separately
Fj_acdf, j_samp_acdf = classical_sampler(
    n_samp_acdf,
    acdf_max_dft_order,
    rescaled_energy_acc,
    nmesh=nmesh,
)
Z_samp_acdf = quantum_sampler(j_samp_acdf, state, hamiltonian, energy_rescalor=None)

# evaluate the ACDF
# acdf_convolution = acdf_kernel(
#     acdf_max_dft_order, Fj_acdf, j_samp_acdf, Z_samp_acdf, energy_grid=energy_grid, nmesh=nmesh
# )
acdf_convolution = acdf_kernel(
    acdf_max_dft_order,
    Fj_acdf,
    j_samp_acdf,
    Z_samp_acdf,
    energy_grid=energy_grid,
    nmesh=nmesh,
)

# Gaussian derivative algorithm

# Perform the classical and quantum sampling separately
Fj_gauss, j_samp_gauss = gaussian_derivative_classical_sampler(
    n_samp_gauss,
    gaussian_derivative_max_dft_order,
    energy_gap,
    rescaled_energy_acc,
    nmesh=nmesh,
)
Z_samp_gauss = quantum_sampler(j_samp_gauss, state, hamiltonian, energy_rescalor=None)

# evaluate the ACDF
# gaussian_derivative_convolution = acdf_kernel(
#     gaussian_derivative_max_dft_order, Fj_gauss, j_samp_gauss, Z_samp_gauss, energy_grid=energy_grid, nmesh=nmesh
# )
gaussian_derivative_convolution = gaussian_derivative_kernel(
    gaussian_derivative_max_dft_order,
    Fj_gauss,
    j_samp_gauss,
    Z_samp_gauss,
    energy_grid=energy_grid,
    nmesh=nmesh,
)

plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1, alpha=0.7)
# Plot the ACDF
acdf_scale = np.linalg.norm(acdf_convolution)
print("acdf scale is:", acdf_scale)
plt.plot(
    energy_grid,
    acdf_convolution / acdf_scale,
    color="blue",
    label="LT22",
    alpha=0.45,
)
gaussian_scale = np.linalg.norm(gaussian_derivative_convolution)
print("gaussian scale is:", gaussian_scale)
plt.plot(
    energy_grid,
    (1 / 4) * gaussian_derivative_convolution / gaussian_scale,
    color="green",
    label="This work",
)
plt.xticks(ticks=[-1.045, 1.045], labels=[r"E$_0$", r"E$_1$"])
# plt.axvline(x=ew[0], color="black", linestyle="-", linewidth=1, alpha=0.7)
# plt.axvline(x=ew[1], color="black", linestyle="-", linewidth=1, alpha=0.7)
plt.yticks(ticks=[0, p0 / 16], labels=["0", r"$\eta$"])
plt.xlabel("Energy")
plt.ylabel("Convolution function")
plt.legend()
# plt.plot(energy_grid, np.zeros_like(energy_grid))
# plt.savefig("figures/gaussian_derivative_convolution.png", dpi=300)

# Plot the Fj_gauss
# plt.plot(
#     np.arange(-gaussian_derivative_max_dft_order, gaussian_derivative_max_dft_order + 1),
#     np.imag(Fj_gauss),
#     linestyle="",
#     marker="o",
#     markersize=3,
# )
# plt.plot(
#     np.arange(-gaussian_derivative_max_dft_order, gaussian_derivative_max_dft_order + 1),
#     np.real(Fj_gauss),
#     linestyle="",
#     marker="o",
#     markersize=3,
# )
# plt.xlabel(r"$index$")
# plt.ylabel(r"$|F_j|$")
# plt.yscale("log")

plt.show()
