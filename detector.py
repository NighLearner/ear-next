import numpy as np
import scipy.io.wavfile as wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import librosa
import os
import gc
from scipy.signal import resample

# Define file paths in Colab's /content/ directory
siren_path = '/content/police_siren.wav'
traffic_path = '/content/Mumbai _Traffic.wav'

# Check if files exist
if not os.path.exists(siren_path) or not os.path.exists(traffic_path):
    raise FileNotFoundError(f"One or both files not found: {siren_path}, {traffic_path}")

# Load audio files (limit to first 10 seconds)
target_fs = 22050  # Downsample to 22.05 kHz
max_duration = 10  # Seconds

try:
    fs_siren, siren = wavfile.read(siren_path)
    fs_traffic, traffic = wavfile.read(traffic_path)
except Exception as e:
    raise ValueError(f"Error loading audio files: {e}")

# Ensure same sampling rate by resampling
if fs_siren != fs_traffic or fs_siren != target_fs:
    # Resample siren
    if fs_siren != target_fs:
        num_samples = int(len(siren) * target_fs / fs_siren)
        siren = resample(siren, num_samples)
        fs_siren = target_fs
    # Resample traffic
    if fs_traffic != target_fs:
        num_samples = int(len(traffic) * target_fs / fs_traffic)
        traffic = resample(traffic, num_samples)
        fs_traffic = target_fs

# Trim to max_duration
max_samples = int(max_duration * target_fs)
min_len = min(len(siren), len(traffic), max_samples)
siren = siren[:min_len]
traffic = traffic[:min_len]

# Convert to mono if stereo
if siren.ndim > 1:
    siren = siren.mean(axis=1)
if traffic.ndim > 1:
    traffic = traffic.mean(axis=1)

# Normalize audio
siren = siren / np.max(np.abs(siren))
traffic = traffic / np.max(np.abs(traffic))

# Create mixed signals (simulate two microphones)
mix_1 = 0.6 * siren + 0.4 * traffic
mix_2 = 0.4 * siren + 0.6 * traffic
X = np.array([mix_1, mix_2])  # Shape: (n_channels, n_samples)

# Clear memory
del siren, traffic
gc.collect()

# Compute STFT with reduced parameters
n_fft = 1024  # Reduced from 2048
hop_length = 256  # Reduced from 512
X_stft = np.array([librosa.stft(x, n_fft=n_fft, hop_length=hop_length) for x in X])
X_stft = np.transpose(X_stft, (2, 1, 0))  # Shape: (n_frames, n_freq, n_channels)

# Apply AuxIVA with fewer iterations
Y_stft = pra.bss.auxiva(X_stft, n_iter=10)

# Clear memory
del X, X_stft
gc.collect()

# Inverse STFT
Y = np.array([
    librosa.istft(Y_stft[:, :, i].T, hop_length=hop_length, length=min_len)
    for i in range(Y_stft.shape[2])
])

# Save separated audio
wavfile.write('/content/separated_siren.wav', target_fs, Y[0])
wavfile.write('/content/separated_traffic.wav', target_fs, Y[1])

# Simple emergency sound detection
siren_spectrum = np.abs(librosa.stft(Y[0], n_fft=n_fft, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=target_fs, n_fft=n_fft)
siren_power = np.mean(siren_spectrum, axis=1)
if np.any(siren_power[freqs > 1000] > 0.1):  # Adjusted threshold
    print("Emergency sound detected in separated source 1")

# Clear memory
del Y, siren_spectrum
gc.collect()

# Plot spectrograms
plt.figure(figsize=(12, 8))

# Mixed signal spectrogram
plt.subplot(3, 1, 1)
librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(Y_stft[:, :, 0].T)),
    sr=target_fs, hop_length=hop_length, x_axis='time', y_axis='hz'
)
plt.title('Mixed Signal (Channel 1)')
plt.colorbar(format='%+2.0f dB')

# Separated siren spectrogram
plt.subplot(3, 1, 2)
librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(Y_stft[:, :, 0].T)),
    sr=target_fs, hop_length=hop_length, x_axis='time', y_axis='hz'
)
plt.title('Separated Siren')
plt.colorbar(format='%+2.0f dB')

# Separated traffic spectrogram
plt.subplot(3, 1, 3)
librosa.display.specshow(
    librosa.amplitude_to_db(np.abs(Y_stft[:, :, 1].T)),
    sr=target_fs, hop_length=hop_length, x_axis='time', y_axis='hz'
)
plt.title('Separated Traffic')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.savefig('/content/bss_results.png')
plt.show()

# Clear memory
del Y_stft
gc.collect()