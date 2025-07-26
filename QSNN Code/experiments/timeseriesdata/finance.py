import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit, Aer, execute
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Constants
NUM_QUBITS = 8
SHOTS = 1024

def load_finance_data(filepath):
    df = pd.read_excel(filepath, parse_dates=['Date'])
    if df['Close*'].dtype == 'object':
        df['Close*'] = pd.to_numeric(df['Close*'].str.replace(',', ''), errors='coerce')
    else:
        df['Close*'] = pd.to_numeric(df['Close*'], errors='coerce')
    df.dropna(subset=['Close*'], inplace=True)
    return df['Close*'].values


def lif_dynamics(x_t, v_prev, alpha=0.9, W=1.0, omega=0.5, k=5.0):
    v_t = alpha * v_prev + W * x_t
    mu_t = 1 if v_t > omega else 0
    if mu_t:
        lam = k * abs(x_t)
        N_spikes = np.random.poisson(lam)
        inter_times = np.random.exponential(1 / (lam + 1e-6), N_spikes)
        spike_times = np.cumsum(inter_times)
        spike_amplitudes = np.random.poisson(lam, N_spikes)
        return spike_times, spike_amplitudes
    return [], []

def bin_spikes(spike_times, spike_amps, Q):
    if len(spike_times) == 0:
        return np.zeros(Q), np.zeros(Q)
    max_t = np.max(spike_times)
    bin_indices = np.floor(Q * spike_times / (max_t + 1e-6)).astype(int)
    bin_indices = np.clip(bin_indices, 0, Q - 1)
    a, tau = np.zeros(Q), np.zeros(Q)
    for i in range(Q):
        mask = bin_indices == i
        if np.any(mask):
            a[i] = np.mean(spike_amps[mask])
            tau[i] = np.mean(spike_times[mask])
    a = np.pi * a / (np.max(a) + 1e-6)
    tau = 2 * np.pi * tau / (np.max(tau) + 1e-6)
    return a, tau

# Single or double encoding
def quantum_encode(amplitudes, times=None, double_encoding=True):
    qc = QuantumCircuit(NUM_QUBITS)
    for j in range(NUM_QUBITS):
        amp = amplitudes[j] if j < len(amplitudes) else 0
        if double_encoding and times is not None:
            phi = times[j] if j < len(times) else 0
            qc.u(amp, phi, -phi, j)
        else:
            qc.rx(amp, j)
    return qc


def apply_vqc(qc, theta):
    for j in range(NUM_QUBITS):
        qc.ry(theta[j], j)
        qc.rz(theta[j + NUM_QUBITS], j)
    for j in range(NUM_QUBITS - 1):
        qc.cx(j, j + 1)
    return qc

def get_measurement_vector(qc):
    qc.measure_all()
    backend = Aer.get_backend('aer_simulator')
    result = execute(qc, backend, shots=SHOTS).result()
    counts = result.get_counts()
    vec = np.array([counts.get(bin(i)[2:].zfill(NUM_QUBITS), 0) for i in range(2 ** NUM_QUBITS)])
    return vec / np.sum(vec)


def encode_sample(x_t, double_encoding=True, theta=None):
    v = 0
    spikes, times = lif_dynamics(x_t, v)
    a, tau = bin_spikes(np.array(times), np.array(spikes), NUM_QUBITS)
    qc = quantum_encode(a, tau if double_encoding else None, double_encoding=double_encoding)
    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi, 2 * NUM_QUBITS)
    qc = apply_vqc(qc, theta)
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
    filepath = "C:\\Users\\riakh\\Downloads\\archive (14)\\yahoo_data.xlsx"
    close_values = load_finance_data(filepath)

    scaler = MinMaxScaler()
    close_values_normalized = scaler.fit_transform(close_values.reshape(-1, 1)).flatten()

    quantum_features, targets = [], []
    for i in range(len(close_values_normalized) - 1):
        encoded = encode_sample(close_values_normalized[i], double_encoding=True)  # Change to False for single
        quantum_features.append(encoded)
        targets.append(close_values_normalized[i + 1])

    X = np.array(quantum_features).reshape((-1, 1, 2**NUM_QUBITS))
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((X.shape[1], X.shape[2]))
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=1)

    loss, mae = model.evaluate(X_test, y_test)
    print("Test Loss:", loss, "Test MAE:", mae)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
