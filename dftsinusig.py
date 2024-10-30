
import numpy as np
import matplotlib.pyplot as plt

# Generate a sinusoidal signal with 1000 samples
N = 1000  # Number of samples
t = np.linspace(0, 1, N, endpoint=False)  # Time vector

# Sinusoidal signal with a known frequency
true_freq = 200 # Frequency of the sinusoid (in Hz)
signal = np.cos(2 * np.pi * 200 * t)

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

# Calculate magnitude spectrum
magnitude = np.abs(dft_result)

# Find the frequency bin with the peak magnitude
peak_index = np.argmax(magnitude)
peak_frequency = peak_index * (1 / N)  # Frequency corresponding to the peak bin

# Plotting the signal and DFT
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title("Original Sinusoidal Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.stem(np.arange(N), magnitude, basefmt=" ")
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")

# Highlight the peak frequency in the magnitude spectrum
plt.subplot(3, 1, 3)
plt.stem(np.arange(N), magnitude, basefmt=" ")
plt.title(f"Peak Frequency at Bin {peak_index} (Frequency = {peak_frequency:.2f} Hz)")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.axvline(x=peak_index, color='r', linestyle='--', label=f'Peak at bin {peak_index}')
plt.legend()

plt.tight_layout()
plt.show()

# Print out the peak frequency and the corresponding bin
print(f"Peak index: {peak_index}")
print(f"Corresponding frequency: {peak_frequency:.2f} Hz")
