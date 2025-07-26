import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, Aer, execute
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

NUM_QUBITS = 8
SHOTS = 1024

def load_temperature_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
    df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')
    df.dropna(inplace=True)
    return df['Daily minimum temperatures'].values

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
    v_prev = 0
    spike_times, spike_amps = lif_dynamics(x_t, v_prev)
    a, tau = bin_spikes(np.array(spike_times), np.array(spike_amps), Q)
    qc = quantum_encode(a, tau)

    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi, 2 * Q)

    qc = apply_vqc_layer(qc, theta)
    return get_measurement_vector(qc)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    filepath = "C:\\Users\\riakh\\Downloads\\daily-minimum-temperatures-in-me.csv"
    temps = load_temperature_data(filepath)

    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps.reshape(-1, 1)).flatten()

    features = []
    targets = []
    for i in range(len(temps_scaled) - 1):
        encoded = encode_sample(temps_scaled[i])
        features.append(encoded)
        targets.append(temps_scaled[i + 1])

    X = np.array(features).reshape(-1, 1, 2 ** NUM_QUBITS)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100,
                        callbacks=[early_stop, reduce_lr], verbose=1)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f} | Test MAE: {mae:.4f}")

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
