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
FC = 433.7e6
# Number of averages
N_AVG = 20
# number of samples
N_SAMP = 100000
# Name of file to save SDR data into
CAPTURE_NAME = 'captures/remote.bin'
# FFT length
N_FFT = 2 ** 12


# Get time data from SDR
time_data = get_time_data(CAPTURE_NAME, FC, FS, N_SAMP, GAIN, use_sdr=True)
raw = np.real(time_data)

# abs and lowpass filter
pos = np.abs(raw)
filt = signal.firwin(numtaps=129, cutoff=5e2, fs=FS)
smooth = np.convolve(pos, filt, mode='same')

# State machine
sig_min = 30
zero_time_min = 100
one_time_min  = 300
in_packet = False
one_count  = 0
zero_count = 0
packet = ''
codes = []
for samp in smooth:
    # Wait until start of packet
    if samp > sig_min:
        in_packet = True
    
    # Packet has started here
    if in_packet:
        # Accumulate time that signal is high
        if samp > sig_min:
            zero_count = 0
            one_count += 1
        # IF end of high pulse, check 1 or 0
        elif samp <= sig_min and one_count != 0:
            if one_count > one_time_min:
                packet += '1'
            elif one_count > zero_time_min:
                packet += '0'
            one_count = 0
        # zero count increment
        elif samp <= sig_min:
            one_count = 0
            zero_count += 1
        
        # Check for end of packet
        if zero_count > one_time_min * 5:
            codes.append(hex(int(packet, 2)))
            packet = ''
            in_packet = False

print(codes)