import numpy as np
import scipy.io.wavfile as wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import librosa

# Load audio files
fs_siren, siren = wavfile.read('siren.wav')
fs_traffic, traffic = wavfile.read('traffic.wav')

# Ensure same sampling rate and length
min_len = min(len(siren), len(traffic))
siren = siren[:min_len]
traffic = traffic[:min_len]
if fs_siren != fs_traffic:
    raise ValueError("Sampling rates must match")

# Normalize audio
siren = siren / np.max(np.abs(siren))
traffic = traffic / np.max(np.abs(traffic))

# Create mixed signals (simulate two microphones)
mix_1 = 0.6 * siren + 0.4 * traffic
mix_2 = 0.4 * siren + 0.6 * traffic
X = np.array([mix_1, mix_2])  # Shape: (n_channels, n_samples)

# Compute STFT
n_fft = 2048
hop_length = 512
X_stft = np.array([librosa.stft(x, n_fft=n_fft, hop_length=hop_length) for x in X])
X_stft = np.transpose(X_stft, (2, 1, 0))  # Shape: (n_frames, n_freq, n_channels)

# Apply AuxIVA
Y_stft = pra.bss.auxiva(X_stft, n_iter=20)

# Inverse STFT to get time-domain signals
Y = np.array([
    librosa.istft(Y_stft[:, :, i].T, hop_length=hop_length, length=min_len)
    for i in range(Y_stft.shape[2])
])

# Save separated audio
wavfile.write('separated_siren.wav', fs_siren, Y[0])
wavfile.write('separated_traffic.wav', fs_siren, Y[1])

# Simple emergency sound detection
siren_spectrum = np.abs(librosa.stft(Y[0], n_fft=n_fft, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=fs_siren, n_fft=n_fft)
siren_power = np.mean(siren_spectrum, axis=1)
if np.any(siren_power[freqs > 1000] > 0.1):  # Check high-frequency power
    print("Emergency sound detected in separated source 1")

# Plot spectrograms
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_stft[:, :, 0].T)), sr=fs_siren, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.title('Mixed Signal (Channel 1)')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(Y_stft[:, :, 0].T)), sr=fs_siren, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.title('Separated Siren')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(Y_stft[:, :, 1].T)), sr=fs_siren, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.title('Separated Traffic')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.savefig('bss_results.png')