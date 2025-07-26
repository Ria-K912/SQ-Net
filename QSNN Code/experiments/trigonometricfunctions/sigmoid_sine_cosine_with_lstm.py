import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


NUM_QUBITS = 5
SHOTS = 1024

def generate_spikes_from_signal(signal):
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 5 + 1
    spikes = np.random.poisson(normalized_signal)
    return spikes

def get_signal_and_time(num_samples, func_type="sigmoid"):
    t = np.linspace(0, 2 * np.pi, num_samples)
    if func_type == "sigmoid":
        signal = 1 / (1 + np.exp(-np.linspace(-6, 6, num_samples)))
    elif func_type == "sine":
        signal = np.sin(t)
    elif func_type == "cosine":
        signal = np.cos(t)
    else:
        raise ValueError("Unsupported signal type")
    return signal, t

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
    for j in range(NUM_QUBITS - 1):
        qc.cx(j, j + 1)
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
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_pipeline(func_type):
    num_samples = 100
    signal, t = get_signal_and_time(num_samples, func_type)
    spikes = generate_spikes_from_signal(signal)

    features = []
    for i in range(num_samples):
        feature = encode_sample(spikes, t)
        features.append(feature)

    features = np.array(features)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features).reshape((num_samples, 1, -1))

    model = build_lstm_model((1, features.shape[1]))
    model.fit(features_scaled, signal, epochs=300, batch_size=1, verbose=1)
    predicted = model.predict(features_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, label='Original Signal', color='blue')
    plt.plot(t, predicted.flatten(), '--', label='Predicted Signal', color='orange')
    plt.title(f'Quantum-LSTM for {func_type.capitalize()} Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


run_pipeline("sigmoid")
run_pipeline("sine")
run_pipeline("cosine")
