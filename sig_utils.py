import os
import numpy as np
import matplotlib.pyplot as plt

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
	with open(filename, 'rb') as f:
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

def view_spectrum(filename, f_center, f_sample, n_samples, gain, n_avg=1, n_fft=4096, use_sdr=True):
	# Get sdr data. We will gather a large time slice that we will break up for FFTs later
	time_data = get_time_data(filename, f_center, f_sample, n_samples * n_avg, gain, use_sdr=use_sdr)
	# Break reshape as matrix to optimize many np.fft operations
	time_data = np.resize(time_data, (n_avg, n_samples))

	# Take N_AVG FFTs 
	fft_mag_avg = np.mean(np.abs(np.fft.fftshift(np.fft.fft(time_data, n=n_fft, axis=1), axes=1)), axis=0)
	fft_axis = np.linspace(f_center - f_sample/2, f_center + f_sample/2, n_fft) / 1e6
		
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

def my_specgram(x, R, L, w, N, Fs):
	sig_len = len(x)
	X = np.zeros((N, sig_len // R))
	for r in range(sig_len // R):
		upper = r * R + L // 2
		lower = r * R - L // 2
		if (upper > sig_len):
			y = np.concatenate((x[lower:], x[:upper % sig_len])) * w
		elif (lower < 0):
			y = np.concatenate((x[lower % sig_len:], x[:upper])) * w
		else:
			y = x[lower:upper] * w
		X[::-1, r] = np.log(np.abs(np.fft.fftshift(np.fft.fft(y, n=N))) + 1e-5)

	plt.imshow(X, interpolation='none', extent=[0,(sig_len)/Fs,-Fs / 2e3,Fs / 2e3], aspect='auto')
	plt.ylabel('Frequency(KHz)')
	plt.xlabel('Time(sec)')
	plt.show()
