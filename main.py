import time
import numpy as np
import sounddevice as sd
from scipy import signal
from sig_utils import *

# SDR Gain (dB)
GAIN = 12.5
# Sample rate (also bandwidth)
FS = 1.024e6
# Center frequency 
FC = 97.300e6
# Number of averages
N_AVG = 20
# number of samples
N_SAMP = 5000000
# Name of file to save SDR data into
CAPTURE_NAME = 'captures/test.bin'
# FFT length
N_FFT = 2 ** 12


# Get time data from SDR
time_data = get_time_data(CAPTURE_NAME, FC, FS, N_SAMP, GAIN, use_sdr=True)

# down sample from 1000KHz to 200KHz
M = 5
filt = signal.firwin(numtaps=129, cutoff=FS/M/2, fs=FS)
time_data = np.convolve(time_data, filt, mode='same')
time_data = time_data[::M]
FS = FS // M

# Convert to audio data
mono_audio = time_data[:-1] * np.conj(time_data[1:])
mono_audio = np.angle(mono_audio)

# down sample from 200KHz to 30KHz
M = 6
filt = signal.firwin(numtaps=129, cutoff=FS/M/2, fs=FS)
mono_audio = np.convolve(mono_audio, filt, mode='same')
mono_audio = mono_audio[::M]
FS = FS // M

# write to file
scaled = mono_audio/np.max(np.abs(mono_audio)) / 50
sd.play(scaled, FS)
time.sleep(len(scaled) / FS)
