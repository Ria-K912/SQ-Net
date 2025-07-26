import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, Aer, execute


NUM_QUBITS = 5
SHOTS = 1024

def generate_spikes_step(num_samples):
    t = np.linspace(0, 1, num_samples)
    signal = np.where(t > 0.5, 1, 0)  # Step function
    spikes = np.random.poisson(signal * 10)  # Poisson spiking
    return spikes, t, signal

def bin_spikes(spikes, times, Q):
    max_t = np.max(times)
    bin_indices = np.floor(Q * times / (max_t + 1e-6)).astype(int)
    bin_indices = np.clip(bin_indices, 0, Q - 1)

    a = np.zeros(Q)
    tau = np.zeros(Q)

    for i in range(Q):
        bin_times = times[bin_indices == i]
        bin_amps = spikes[bin_indices == i]
        if len(bin_amps) > 0:
            a[i] = np.mean(bin_amps)
            tau[i] = np.mean(bin_times)

    a = np.pi * a / (np.max(a) + 1e-6)
    tau = 2 * np.pi * tau / (np.max(tau) + 1e-6)
    return a, tau

def quantum_encode(amplitudes, times):
    qc = QuantumCircuit(NUM_QUBITS)
    for j in range(NUM_QUBITS):
        amp = amplitudes[j] if j < len(amplitudes) else 0
        phi = times[j] if j < len(times) else 0
        qc.u(amp, phi, -phi, j)
    return qc

def apply_vqc(qc, theta):
    for j in range(NUM_QUBITS):
        qc.ry(theta[j], j)
        qc.rz(theta[j + NUM_QUBITS], j)
    return qc

def encode_sample(spikes, t, theta=None):
    Q = NUM_QUBITS
    a, tau = bin_spikes(spikes, t, Q)
    qc = quantum_encode(a, tau)
    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi, 2 * Q)
    qc = apply_vqc(qc, theta)
    qc.measure_all()
    backend = Aer.get_backend("aer_simulator")
    result = execute(qc, backend, shots=SHOTS).result()
    counts = result.get_counts()
    vec = np.array([counts.get(bin(i)[2:].zfill(Q), 0) for i in range(2 ** Q)])
    return vec / np.sum(vec)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def process_and_plot_step():
    num_samples = 100
    spikes, t, signal = generate_spikes_step(num_samples)

    features = []
    for i in range(num_samples):
        features.append(encode_sample(spikes, t))

    features = np.array(features)

    # Scale
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = features_scaled.reshape((features.shape[0], 1, -1))

    model = build_lstm_model((1, features.shape[1]))
    model.fit(features_scaled, signal, epochs=300, batch_size=10, verbose=1)

    predicted_signal = model.predict(features_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, label='Original Step Signal', color='blue')
    plt.plot(t, predicted_signal.flatten(), '--', label='Predicted Signal', color='orange')
    plt.title('Quantum-LSTM Prediction of Step Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    process_and_plot_step()
