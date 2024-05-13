import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
import scipy.signal as signal

def audio_from_wav(filename):
    fs, data = wavfile.read(filename)
    print(type(data[0,0]))
    audio = (data[:,0] + data[:,1]) // 2
    data[:,0] = audio
    data[:,1] = audio
    wavfile.write("mono_of_" + filename, fs, data)
    return fs, audio;

def audio_to_wav(audio, filename, fs):
    data = np.zeros((len(audio), 2)).astype(np.int16)
    data[:,0] = audio
    data[:,1] = audio
    print(type(data[0,0]))
    wavfile.write(filename, fs, data)

def pad_audio(audio, fs, df):
    N = len(audio)
    N_min = fs/df
    N_req = int((N // N_min + 1) * N_min - N)
    pad = np.linspace(0, 0, N_req)
    audio = np.concatenate((audio, pad))
    return audio

def remove_low_frequencies(audio, fs, f_max):
    audio_fft = scipy.fftpack.rfft(audio)
    audio_fft[0:len(audio)*f_max*2//fs] = 0
    audio = np.real(scipy.fftpack.irfft(audio_fft))
    return audio

def add_data(audio, data, ampl, fs, df, f_min):
    f_data = np.linspace(f_min, f_min + (len(data)-1) * df, len(data))
    f_audio = np.fft.rfftfreq(len(audio), 1/fs)
    audio_fft = np.fft.rfft(audio)

    f_d = 0
    for f in range(len(f_audio)):
        if (f_data[f_d] >= f_audio[f]):
            audio_fft[f] = ampl * data[f_d]
            f_d += 1
            if f_d == len(f_data):
                break

    audio = np.real(np.fft.irfft(audio_fft))
    return audio