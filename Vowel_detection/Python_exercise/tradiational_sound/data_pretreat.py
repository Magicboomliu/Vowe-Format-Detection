
# 工具包
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import librosa
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

def FFT(frames,NFFT):
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    print(pow_frames.shape)
    return  pow_frames

def get_fBank(powd_frames,sameple_rate,NFFT,nfilt):
    '''
    
    :param frames: Frames after NFFT
    :param sameple_rate: 采样率
    :param nift: 规定有多少个mel滤波器 
    :return: FBank Features
    '''
    ''' 规定mel值的上限和下限'''
    low_freq_mel = 0
    # 根据葵姐斯特采样定理可得
    high_freq_mel = 2595 * np.log10(1 + (sameple_rate / 2) / 700)
    # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    # 各个mel滤波器在能量谱对应点的取值
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    bin = (hz_points / (sameple_rate / 2)) * (NFFT / 2)
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    filter_banks = np.dot(powd_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    print(filter_banks.shape)
    return fbank,filter_banks


def get_mfcc_features(num_ceps,filter_banks,lifted=False,cep_lifter=23):
    # 使用DCT，提取2-13维，得到MFCC特征
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    if (lifted):
        # 对Mfcc进行升弦，平滑这个特征
        cep_lifter = 23
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        print(mfcc.shape)
    return mfcc

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_mfcc_from_files(file):
    signal,sample_rate = librosa.load(file,sr=8000)
    # 获取信号的前3.5s
    #signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
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

    #plot_time(windowed_frames[1], sample_rate)
    pow_frames = FFT(windowed_frames,512)

    #经过上面的步骤之后，在能量谱上应用Mel滤波器组，就能提取到FBank特征
    fbank,filter_banks=get_fBank(pow_frames,sample_rate,512,40)

    print("FBank features",filter_banks.shape)
    plot_spectrogram(filter_banks.T, 'Filter Banks')
    mfcc = get_mfcc_features(num_ceps=12,filter_banks=filter_banks,
                             lifted=True)
    print(mfcc.shape)                         
    return filter_banks
    

    # lables = np.zeros(shape=(filter_banks.shape[0]))
    # print(lables.shape)
    # dct ={"sound_mfcc":mfcc,"labels":lables}
    # with open("mfcc_new",'wb') as f1:
    #     pickle.dump(dct,f1)


if __name__ == "__main__":

    file_list=["Vowe_a.wav","Vowe_e.wav","vowe_i.wav","Vowe_o.wav","vowe_u.wav"]
    mfcc_old = get_mfcc_from_files(file_list[0])
    # labels
    lables = np.zeros(shape=(mfcc_old.shape[0]))

    print(lables)
    for idx,file in enumerate(file_list):
        if idx==0:
            continue
        
        mfcc_new = get_mfcc_from_files(file)
        new_label= np.zeros(mfcc_new.shape[0])+idx
        lables =np.hstack((lables,new_label))
        mfcc_old =np.vstack((mfcc_old,mfcc_new))
    
    print(mfcc_old.shape)
    print(lables.shape)

    dct={"MFCC":mfcc_old,"labels":lables}
    with open("mfcc_d",'wb') as f1:
         pickle.dump(dct,f1)





    



