import numpy as np
import scipy
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa
import math
import wave
from scipy.signal import lfilter, hamming
import warnings
warnings.filterwarnings('ignore')
import pickle
import kaldi
from kaldi.feat.plp import Plp,PlpOptions
from kaldi.matrix import SubVector, SubMatrix
from spafe.utils import vis
from spafe.features.rplp import rplp, plp

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
    print("Nums of frames",num_frames)
    print("each frame has orginal frams",frame_length)
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

# get plp or rplp feature from a frame
def get_plps(frame,sample_rate,num_ceps=13,r=False):
    if r ==True:
        plps = rplp(frame, sample_rate, num_ceps)
    else:
        plps =plp(frame, sample_rate, num_ceps)
    return plps

def get_plps_files(filename,r=False):
    signal,sample_rate = librosa.load(filename,sr=8000)
    # 获取信号的前3.5s
    # 显示相关signal的信息
    print('sample rate:', sample_rate, ', frame length:', len(signal))
    # 画出相关的图像
    # plot_time(signal, sample_rate)
    # plot_freq(signal, sample_rate)
    # 预加重,达到平衡频谱的作用.
    pre_emphasis_signal = pre_emphasis_func(signal)
    # plot_time(pre_emphasis_signal, sample_rate)
    # plot_freq(pre_emphasis_signal, sample_rate)
    frames,frame_length,frame_step = Frameing(pre_emphasis_signal,sample_rate,0.015,0.01)
    windowed_frames =Windowing(frames,frame_length)
    plps_first = get_plps(windowed_frames[0],sample_rate,13,r)
    plps_first = plps_first.reshape(1,-1,13)

    for idx, wf in enumerate(windowed_frames):
        if idx==0:
            continue
        # get plps features
        plps_next = get_plps(wf,sample_rate,13,False)
        plps_next =plps_next.reshape(1,-1,13)
        plps_first = np.vstack((plps_first,plps_next))
    
    print(plps_first.shape)
    return plps_first


    
if __name__ == "__main__":

    file_list=["Vowe_a.wav","Vowe_e.wav","vowe_i.wav","Vowe_o.wav","vowe_u.wav"]
    
    plps_old = get_plps_files(file_list[0],r=True)
    labels_old = np.zeros(shape=plps_old.shape[0])

    for idx,filename in enumerate(file_list):
        if idx==0:
            continue
        plps_new = get_plps_files(file_list[idx],r=True)
        plps_old = np.vstack((plps_old,plps_new))

        labels_new = np.zeros(shape=plps_new.shape[0]) + idx
        labels_old = np.hstack((labels_old,labels_new))
    
    print(plps_old.shape)
    print(labels_old.shape)

    dct ={"RPLP":plps_old,"labels":labels_old}
    with open("rplp_t",'wb') as f1:
         pickle.dump(dct,f1)
         print("Write to file Successfully.")

    
    






# print(windowed_frames[0])

# plps = plp(windowed_frames[0], sample_rate, 13)
# print(plps.shape)
# plps2 = plp(windowed_frames[1], sample_rate, 13)
# print(plps2.shape)



