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
# 对每一帧的信号，进行快速傅里叶变换，对于每一帧的加窗信号，进行N点FFT变换，也称短时傅里叶变换（STFT），N通常取256或512，然后用如下的公式计算能量谱
def get_lcp_hz(filename):

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
    windowed_frames = lfilter([1., 0.63], 1, windowed_frames)
    print(len(windowed_frames))

    lpcs_coeff = librosa.lpc(y=windowed_frames[0],order=16)
    sols =np.roots(lpcs_coeff)
   
    # 保留虚部大于0的部分
    roots=[]
    for r in sols:
        if np.imag(r)>0:
            roots.append(r)

    angz = np.arctan2(np.imag(roots), np.real(roots))
    # GeT F1 and F2 and F3
    frqs = sorted(angz * (sample_rate / (2 * math.pi)))
    frqs_old =np.array(frqs[:6]).reshape(1,-1)
    print("Frqs_old shape is ",frqs_old.shape)

    # get lcp coeffs
    for j in range(len(windowed_frames)):
        if j==0:
            continue
        if j==len(windowed_frames)-1:
            break
        lpcs_coeff = librosa.lpc(y=windowed_frames[j],order=16)
        sols =np.roots(lpcs_coeff)
        
        print(j,"/{}".format(len(windowed_frames)))
       
        # 保留虚部大于0的部分
        roots=[]
        for r in sols:
            if np.imag(r)>0:

                roots.append(r)
        angz = np.arctan2(np.imag(roots), np.real(roots))
        # GeT F1 and F2 and F3
        frqs = sorted(angz * (sample_rate / (2 * math.pi)))
        frqs_np = np.array(frqs[:6]).reshape(1,-1)
        print("Frqs_np shape is ",frqs_np.shape)
        frqs_old =np.vstack((frqs_old,frqs_np))
    print(frqs_old.shape)

    return frqs_old
        




if __name__ == "__main__":

    file_list=["Vowe_a.wav","Vowe_e.wav","vowe_i.wav","Vowe_o.wav","vowe_u.wav"]
    frqs_old = get_lcp_hz(file_list[0])
    
    labels_old = np.zeros(shape=frqs_old.shape[0])

    for idx, filename in enumerate(file_list):
        if idx==0:
            continue
        freqs_new = get_lcp_hz(file_list[idx])
        labels_new = np.zeros(shape=freqs_new.shape[0]) +idx
        frqs_old = np.vstack((frqs_old,freqs_new))
        labels_old = np.hstack((labels_old,labels_new))
    
    print(frqs_old.shape)
    print(labels_old.shape)

    dct={"LCPs":frqs_old,"labels":labels_old}
    with open("lcp_t",'wb') as f1:
         pickle.dump(dct,f1)
 