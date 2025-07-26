import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute


NUM_QUBITS = 8


def generate_gaussian_spikes(num_samples=100):
    t = np.linspace(-5, 5, num_samples)
    signal = np.exp(-t**2)  # Gaussian
    normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 5 + 1
    spikes = np.random.poisson(normalized)
    return t, spikes, signal



def bin_spikes(t, spikes, Q=NUM_QUBITS):
    max_t = np.max(t)
    bin_indices = np.floor(Q * (t - np.min(t)) / (max_t - np.min(t) + 1e-6)).astype(int)
    bin_indices = np.clip(bin_indices, 0, Q - 1)

    a = np.zeros(Q)
    tau = np.zeros(Q)
    for i in range(Q):
        bin_times = t[bin_indices == i]
        bin_spike_vals = spikes[bin_indices == i]
        if len(bin_spike_vals) > 0:
            a[i] = np.mean(bin_spike_vals)
            tau[i] = np.mean(bin_times)

    a = np.pi * a / (np.max(a) + 1e-6)
    tau = 2 * np.pi * (tau - np.min(tau)) / (np.max(tau) - np.min(tau) + 1e-6)
    return a, tau


def quantum_encode(a, tau):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(range(NUM_QUBITS))  # Hadamard to all
    for j in range(NUM_QUBITS):
        amp = a[j] if j < len(a) else 0
        phi = tau[j] if j < len(tau) else 0
        qc.u(amp, phi, -phi, j)
    return qc


def apply_vqc_layer(qc, theta):
    for j in range(NUM_QUBITS):
        qc.ry(theta[j], j)
        qc.rz(theta[j + NUM_QUBITS], j)
    for j in range(NUM_QUBITS - 1):
        qc.cx(j, j + 1)
    return qc


def get_statevector(qc):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    return result.get_statevector(qc)


def reconstruct_signal(statevector, t):
    reconstructed = np.zeros_like(t)
    for amp in statevector:
        reconstructed += abs(amp) * np.exp(-t**2) * np.cos(np.angle(amp))
    return reconstructed


def plot_results(t, original_signal, reconstructed_signal):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, original_signal, 'b-', label='Original Gaussian Signal')
    plt.title("Original Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (Quantum)')
    plt.title("Reconstructed Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    t, spikes, original_signal = generate_gaussian_spikes()
    a, tau = bin_spikes(t, spikes)
    qc = quantum_encode(a, tau)
    theta = np.random.uniform(0, 2 * np.pi, 2 * NUM_QUBITS)
    qc = apply_vqc_layer(qc, theta)
    statevector = get_statevector(qc)
    reconstructed_signal = reconstruct_signal(statevector, t)
    plot_results(t, original_signal, reconstructed_signal)

if __name__ == "__main__":
    main()
