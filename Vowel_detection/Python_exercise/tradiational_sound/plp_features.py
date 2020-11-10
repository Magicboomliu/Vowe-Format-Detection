__author__ = "Luke Liu"
#encoding="utf-8"
import scipy
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import librosa
import math
import wave
from scipy.signal import lfilter, hamming
import scipy
from spafe.utils import vis
from spafe.features.rplp import rplp, plp
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')
# init input vars




# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2, fft_size/2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()

# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()

# 预加重
def pre_emphasis_func(signal):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return  emphasized_signal

# Frameing
def Frameing(signal,sample_rate,frame_size,frame_stride):
    #frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,1)
    frames = pad_signal[indices]
    print(frames.shape)
    return frames,frame_length,frame_step

# 让每一帧的2边平滑衰减，这样可以降低后续傅里叶变换后旁瓣的强度，取得更高质量的频谱。
def Windowing(frames,frame_length):
    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))
    windowed_frames =frames*hamming
    return  windowed_frames





    
# read wav 

sample_rate, signal = wavfile.read("123.wav")

# # compute features
# plps = plp(sig, fs, num_ceps)
# rplps = rplp(sig, fs, num_ceps)

# # visualize spectogram
# vis.spectogram(sig, fs)
# # visualize features
# vis.visualize_features(plps, 'PLP Coefficient Index', 'Frame Index')
# vis.visualize_features(rplps, 'RPLP Coefficient Index', 'Frame Index')


if __name__ == "__main__":
    num_ceps = 13
    sample_rate, signal = wavfile.read("123.wav")
    pre_emphasis_signal = pre_emphasis_func(signal)
    # plot_time(pre_emphasis_signal, sample_rate)
    # plot_freq(pre_emphasis_signal, sample_rate)
    frames,frame_length,frame_step = Frameing(pre_emphasis_signal,sample_rate,0.025,0.01)
    windowed_frames =Windowing(frames,frame_length)
    print(windowed_frames.shape)
    plps = plp(windowed_frames[0],sample_rate, num_ceps)
    print(plps.shape)
    