import librosa
import matplotlib.pyplot as plt
import numpy as np

import os, sys
from dotenv import load_dotenv
load_dotenv()

vocals_path = os.getenv("VOCALS_PATH")
song_title = os.getenv("SONG_TITLE")

#get audio timeseries and sampling rate
audio_array, sample_rate = librosa.load(vocals_path, sr=None)

#create a figure with 2 subplots (waveform and spectrogram)
fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
ax[0].set_title(f'Waveform | {song_title}')
ax[1].set_title(f'Spectrogram | {song_title}')

#waveshow - the waveform of the vocals
librosa.display.waveshow(audio_array, sr=sample_rate, ax=ax[0])

#spectrogram - the frequency content of the vocals over time
stft = librosa.stft(audio_array)
db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
librosa.display.specshow(db_spectrogram, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax[1])

#spectral centroid - the "center of mass" of the spectrum, which can indicate brightness
spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
spectral_centroids = spectral_centroids[0] # squeezing to 1D array
frames = range(len(spectral_centroids))
time_frames = librosa.frames_to_time(frames, sr=sample_rate)

#plot the spectral centroid on top of the spectrogram
ax[1].plot(time_frames, spectral_centroids, color='white', linewidth=2, alpha=0.7, label='Spectral Centroid')
ax[1].legend(loc='upper right')

spectral_flatness = librosa.feature.spectral_flatness(y=audio_array)
rms = librosa.feature.rms(y=audio_array)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_array)
spectral_contrast = librosa.feature.spectral_contrast(y=audio_array)

feature_text = f"""
Spectral Flatness: {np.mean(spectral_flatness):.4f}
RMS Energy: {np.mean(rms):.4f}
Zero Crossing Rate: {np.mean(zero_crossing_rate):.4f}
Spectral Contrast: {np.mean(spectral_contrast):.4f}
"""

ax[0].text(0.98, 0.98, feature_text, 
           transform=ax[0].transAxes,  # Use axes coordinates (0-1)
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9,
           family='monospace')


plt.show()