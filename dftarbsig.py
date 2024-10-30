import numpy as np
import matplotlib.pyplot as plt

# Given arbitrary signal
signal = np.array([1, 2, 3, 4])
N = len(signal)  # Length of the signal

# Implementing the DFT
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Perform DFT
dft_result = DFT(signal)

# Calculate magnitude and phase
magnitude = np.abs(dft_result)
phase = np.angle(dft_result)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.stem(np.arange(N), signal, basefmt=" ")
plt.title("Original Signal x[n]")
plt.xlabel("Sample Index n")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.stem(np.arange(N), magnitude, basefmt=" ")
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")

plt.subplot(3, 1, 3)
plt.stem(np.arange(N), phase, basefmt=" ")
plt.title("Phase Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase [radians]")

plt.tight_layout()
plt.show()
