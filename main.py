import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import firwin
import sounddevice as sd

# SDR Gain (dB)
GAIN = 12.5
# Sample rate (also bandwidth)
FS = 1e6
# Center frequency 
FC = 97.3e6
# Number of averages
N_AVG = 20
# number of samples
N_SAMP = 1000000
# Name of file to save SDR data into
CAPTURE_NAME = 'captures/test.bin'
# FFT length
N_FFT = 2 ** 12


def get_time_data(filename, f_center, f_sample, num_samples, gain, use_sdr=True):
	if use_sdr:
		# Call `rtl_sdr` application
		cmd =  'rtl_sdr ' + filename
		cmd += ' -f ' + str(f_center)
		cmd += ' -s ' + str(f_sample)
		cmd += ' -n ' + str(num_samples)
		cmd += ' -g ' + str(gain)
		os.system(cmd)

	# Open SDR data file
	with open(CAPTURE_NAME, 'rb') as f:
		bytes_read = list(f.read())

	# read IQ
	I = np.array(bytes_read[::2]) - 127.5
	Q = np.array(bytes_read[1::2]) - 127.5
	time_data = I + 1j * Q

	# return time data
	return time_data

def plot_fft(time_sig, fs, fc=0, n_fft=4096):
	fft = np.fft.fftshift(np.fft.fft(time_sig, n=n_fft))
	fft_axis = np.linspace(fc + -fs/2,fc + fs/2, n_fft)
	plt.figure()
	plt.plot(fft_axis, np.abs(fft))

def view_spectrum(filename, f_center, f_sample, num_samples, gain, n_avg=1, use_sdr=True):
	# Get sdr data. We will gather a large time slice that we will break up for FFTs later
	time_data = get_time_data(filename, f_center, f_sample, num_samples * n_avg, gain, use_sdr=use_sdr)
	# Break reshape as matrix to optimize many np.fft operations
	time_data = np.resize(time_data, (N_AVG, N_SAMP))

	# Take N_AVG FFTs 
	fft_mag_avg = np.mean(np.abs(np.fft.fftshift(np.fft.fft(time_data, n=N_FFT, axis=1), axes=1)), axis=0)
	fft_axis = np.linspace(FC - FS/2, FC + FS/2, N_FFT) / 1e6
		
	# Plot decibel scale
	plt.figure()
	plt.plot(fft_axis, fft_mag_avg)
	plt.xlabel('MHz')

def get_name_extension(filename):
    idx = len(filename) - 1
    while filename[idx] != '.' and idx > 0:
        idx -= 1
    if idx == 1:
        print('Error with filename')
        return
    name = filename[:idx]
    extension = filename[idx:]

    return name, extension

# # Show the spectrum with nice averages
# view_spectrum(CAPTURE_NAME, FC, FS, N_SAMP, GAIN, N_AVG, use_sdr=True)

# Get time data from SDR
time_data = get_time_data(CAPTURE_NAME, FC, FS, N_SAMP, GAIN, use_sdr=True)

# Low pass filter
filt = firwin(numtaps=129, cutoff=100e3, fs=FS)
#time_data = np.convolve(time_data, filt, mode='same')

# down sample
M = 5
time_data = time_data[::M]
time_data = time_data / (np.abs(time_data) + 1e-6)

# Convert to audio data
phase = np.angle(time_data)
deriv = np.diff(phase)

# write to file
scaled = deriv/np.max(np.abs(deriv)) / 50
sd.play(scaled, FS // M // 10)
plt.plot(scaled)
plt.show()
