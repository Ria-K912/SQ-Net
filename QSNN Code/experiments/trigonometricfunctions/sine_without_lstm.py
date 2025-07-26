import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute


NUM_QUBITS = 8
FREQUENCY = 5


def generate_spike_train(num_samples=100, frequency=FREQUENCY):
    t = np.linspace(0, 1, num_samples)
    signal = np.sin(2 * np.pi * frequency * t)
    poisson_rate = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 5 + 1
    spikes = np.random.poisson(poisson_rate)
    return t, spikes, signal


def bin_spikes(t, spikes, Q=NUM_QUBITS):
    max_t = np.max(t)
    bin_indices = np.floor(Q * t / (max_t + 1e-6)).astype(int)
    bin_indices = np.clip(bin_indices, 0, Q - 1)

    a = np.zeros(Q)
    tau = np.zeros(Q)

    for i in range(Q):
        bin_times = t[bin_indices == i]
        bin_amps = spikes[bin_indices == i]
        if len(bin_amps) > 0:
            a[i] = np.mean(bin_amps)
            tau[i] = np.mean(bin_times)

    
    a = np.pi * a / (np.max(a) + 1e-6)
    tau = 2 * np.pi * tau / (np.max(tau) + 1e-6)
    return a, tau


def quantum_encode(amplitudes, phases):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(range(NUM_QUBITS))  # Hadamard initialization
    for j in range(NUM_QUBITS):
        amp = amplitudes[j] if j < len(amplitudes) else 0
        phi = phases[j] if j < len(phases) else 0
        qc.u(amp, phi, -phi, j)  # U(a, τ, -τ)
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


def reconstruct_signal_from_state(statevector, t, frequency=FREQUENCY):
    reconstructed = np.zeros_like(t)
    for amp in statevector:
        reconstructed += abs(amp) * np.sin(2 * np.pi * frequency * t + np.angle(amp))
    return reconstructed


def main_signal_reconstruction():
    t, spikes, original_signal = generate_spike_train()
    a, tau = bin_spikes(t, spikes)
    qc = quantum_encode(a, tau)

    theta = np.random.uniform(0, 2 * np.pi, 2 * NUM_QUBITS)
    qc = apply_vqc_layer(qc, theta)

    statevector = get_statevector(qc)
    reconstructed = reconstruct_signal_from_state(statevector, t)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, original_signal, label="Original Sine Wave")
    plt.plot(t, reconstructed, '--', label="Reconstructed from Quantum State")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal Reconstruction via Spike-based Quantum Encoding + VQC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_signal_reconstruction()
