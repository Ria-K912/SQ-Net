import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


NUM_QUBITS = 8  
NUM_SAMPLES = 100
SHOTS = 1024

def load_mackey_glass_data(filepath):
    df = pd.read_excel(filepath)
    return df['t'].values, df['t-taw'].values, df['t+1'].values

def lif_dynamics(x_t, v_prev, alpha=0.9, W=1.0, omega=0.5, k=5.0):
    v_t = alpha * v_prev + W * x_t
    mu_t = 1 if v_t > omega else 0
    if mu_t == 1:
        lam = k * abs(x_t)
        N_spikes = np.random.poisson(lam)
        inter_times = np.random.exponential(1 / (lam + 1e-6), N_spikes)
        spike_times = np.cumsum(inter_times)
        spike_amplitudes = np.random.poisson(lam, N_spikes)
        return spike_times, spike_amplitudes
    return [], []

def bin_spikes(spike_times, spike_amplitudes, Q):
    if len(spike_times) == 0:
        return np.zeros(Q), np.zeros(Q)
    max_t = np.max(spike_times)
    bin_indices = np.floor(Q * spike_times / (max_t + 1e-6)).astype(int)
    bin_indices = np.clip(bin_indices, 0, Q - 1)

    a = np.zeros(Q)
    tau = np.zeros(Q)

    for i in range(Q):
        bin_times = spike_times[bin_indices == i]
        bin_amps = spike_amplitudes[bin_indices == i]
        if len(bin_amps) > 0:
            a[i] = np.mean(bin_amps)
            tau[i] = np.mean(bin_times)
    a = np.pi * a / (np.max(a) + 1e-6)
    tau = 2 * np.pi * tau / (np.max(tau) + 1e-6)
    return a, tau

def quantum_encode(amplitudes, times):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(range(NUM_QUBITS))

    for j in range(NUM_QUBITS):
        amp = amplitudes[j] if j < len(amplitudes) else 0
        phi = times[j] if j < len(times) else 0
        qc.u(amp, phi, -phi, j)

    return qc


def apply_vqc_layer(qc, params):
    for j in range(NUM_QUBITS):
        qc.ry(params[j], j)
        qc.rz(params[j + NUM_QUBITS], j)
        
    for j in range(NUM_QUBITS - 1):
        qc.cx(j, j + 1)

    return qc


def get_measurement_vector(qc, shots=SHOTS):
    qc.measure_all()
    backend = Aer.get_backend('aer_simulator')
    result = execute(qc, backend, shots=shots).result()
    counts = result.get_counts()
    vec = np.array([counts.get(bin(i)[2:].zfill(NUM_QUBITS), 0) for i in range(2 ** NUM_QUBITS)])
    return vec / np.sum(vec)

def encode_sample(x_t, Q=NUM_QUBITS, theta=None):
    v = 0
    spikes, times = lif_dynamics(x_t, v)
    a, tau = bin_spikes(np.array(times), np.array(spikes), Q)
    qc = quantum_encode(a, tau)

    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi, 2 * Q)
    qc = apply_variational_layer(qc, theta)

    return get_measurement_vector(qc)

def build_hybrid_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    t, t_taw, t_plus_1 = load_mackey_glass_data("C:\\Users\\riakh\\Downloads\\archive (13)\\Mackey-Glass Time Series(taw17).xlsx")
    t_norm = (t - np.min(t)) / (np.max(t) - np.min(t))
    t_taw_norm = (t_taw - np.min(t_taw)) / (np.max(t_taw) - np.min(t_taw))
    
    features = []
    targets = []
    for xt, yt in zip(t_taw_norm[:-1], t_plus_1[:-1]):
        encoded = encode_sample(xt)
        features.append(encoded)
        targets.append(yt)

    X = np.array(features)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_hybrid_model(X.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=1)

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss, "Test MAE:", test_mae)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
