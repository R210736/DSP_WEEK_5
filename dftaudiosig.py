import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the speech waveform
fs, speech = wavfile.read('/home/sumanth/Classical.wav')

# Perform DFT
dft = np.fft.fft(speech)

# Calculate frequency axis
freq = np.fft.fftfreq(len(speech), d=1/fs)

# Plot the speech waveform
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(speech)
plt.title("Speech Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot the DFT
plt.subplot(3, 1, 2)
plt.plot(np.abs(dft))
plt.title("DFT of Speech Waveform")
plt.xlabel("Frequency Index")
plt.ylabel("Magnitude")

# Plot the frequency plot varying
plt.subplot(3, 1, 3)
plt.plot(freq, np.abs(dft))
plt.title("Frequency Plot Varying")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()