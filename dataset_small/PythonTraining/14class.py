import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import librosa
import pickle
from PixelShift.explore_data import PixelShiftSound
# SVM libraies
from sklearn import svm
from sklearn import preprocessing
import sklearn.metrics
import pickle

'''可视化： 混淆矩阵'''
def plot_confusion_matrix(cm, labels_name, title,figname):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig(figname, format='png')
    plt.show()
    

'''预加重'''
# 首先对数据进行预加重
def pre_emphasis_func(signal):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return  emphasized_signal
'''窗口化'''
# 让每一帧的2边平滑衰减，这样可以降低后续傅里叶变换后旁瓣的强度，取得更高质量的频谱。
def Windowing(frames,frame_length):
    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))
    windowed_frames =frames*hamming
    return  windowed_frames
''' 傅里叶变换'''
# 对每一帧的信号，进行快速傅里叶变换，对于每一帧的加窗信号，进行N点FFT变换，也称短时傅里叶变换（STFT），N通常取256或512，然后用如下的公式计算能量谱
def FFT(frames,NFFT):
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    print(pow_frames.shape)
    return  pow_frames

'''fank特征 40 dim'''
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

''' 获取MFCC特征'''
def get_mfcc_features(num_ceps,filter_banks,lifted=False,cep_lifter=23):
    # 使用DCT，提取2-13维，得到MFCC特征
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    if (lifted):
        # 对Mfcc进行升弦，平滑这个特征
        cep_lifter = 23
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        
    return mfcc


if __name__ == "__main__":

    '''建立标签index与Viseme的映射关系'''
    category_mapping ={0:"sil",1:"PP",2:"FF",3:"TH",4:"DD",5:"kk",6:"CH",7:"SS",8:"nn",9:"RR",
    10:"aa",11:"E",12:"ih",13:"oh",14:"ou"}

    '''得到数据集中的音频域与标签数据，并将其转换为mfcc特征'''
    ps = PixelShiftSound(sample_rate=16000,frame_duration=0.016,frame_shift_duration=0.008,datatype=2)
    wav_data,wav_label = ps.get_all_wav_data()
    nums_of_data = wav_data.shape[0]
  
    print("Wav Frame data：",wav_data.shape)
    print("Wav Frame label：",wav_label.shape)
    for i in range(len(wav_data)):
        # 预加重和窗口化处理
        wav_data[i] = pre_emphasis_func(wav_data[i])
    
    wav_data = Windowing(wav_data,len(wav_data[0]))
    # 对每一帧进行傅里叶变换
    fft_data = FFT(wav_data,512)
    print("Wav Frame data After FFT：",wav_data.shape)
    fbank,filter_banks=get_fBank(fft_data,16000,512,40)
    print("Wav Frame data After FBanks：",wav_data.shape)
    # 到这里都是OK的
    mfcc_data = get_mfcc_features(num_ceps=12,filter_banks=filter_banks,
                             lifted=True)
       
    print("Wav Frams's MFCC features",mfcc_data.shape)
    
    
    
    
    # # 方法一： 当sil 不为1的时候，去掉sil取第二大作为分类结果
    # '''得到label的特征，取最大值的下标作为index'''
    # cnt = 0
    # saved_index =[]
    # label_list=[]
    # for index,vector in enumerate(wav_label):
    #     lst = vector.tolist()
    #     if (lst[0]==1):
    #         continue
    #     else:
    #        saved_index.append(index)
    #        list_a_max_list = max(lst[1:]) #返回最大值
    #        max_index = lst[1:].index(list_a_max_list) # 返回最大值的索引
    #        label_list.append(max_index)
    #方法2 :直接去掉所有sil为1的点， 对sil不为1的点进行14分类
    cnt =0
    saved_index =[]
    label_list=[]
    for index,vector in enumerate(wav_label):
        lst = vector.tolist()
        list_a_max_list = max(lst) # 返回最大值
        max_index = lst.index(list_a_max_list)
        if max_index==0:
            continue
        else:
            saved_index.append(index)  # 记录当前是第几条数据
            label_list.append(max_index-1)  # 记录当前最大的数据是多少
           

    label_list = np.array(label_list)
    print("取最大值后的wav label data:", label_list.shape)
    nums_new = len(saved_index)
    new_mfcc_data = np.zeros((nums_new,mfcc_data.shape[1]))
    for i,index in  enumerate(saved_index):
        new_mfcc_data[i] = mfcc_data[index]
    
    print("取最大值后的wav data:", new_mfcc_data.shape)


    ''' 准备利用SVM进行分类 '''
    # 首先对数据进行Normalization
    trainig_data = preprocessing.scale(new_mfcc_data)
    
    # 对数据进行shuffer操作，混乱化
    state = np.random.get_state()
    np.random.shuffle(trainig_data)
    np.random.set_state(state)
    np.random.shuffle(label_list)
    

     #Save into a pickle file
    dct={"MFCC":trainig_data,"labels":label_list}
    with open("mfcc14_RIGHT",'wb') as f1:
         pickle.dump(dct,f1)
    
    print("DONE")
    print(len(set(label_list)))
    
    trains_nums = int(trainig_data.shape[0]*0.8)
    val_nums = int(trainig_data.shape[0] * 0.2)
    trains_d = trainig_data[0:trains_nums]
    trains_l = label_list[0:trains_nums]

    val_d = trainig_data[trains_nums:val_nums+trains_nums]
    val_l = label_list[trains_nums:val_nums+trains_nums]

    
    # 建立一个SVM classifier
    # Using RBF as Kernel Functions
    
    for i in range(1,11):
        print("Start Training")
        classifier =svm.SVC(C=i,kernel='rbf',gamma='auto',decision_function_shape='ovr')
        classifier.fit(trains_d,trains_l.ravel())
        print("Training is Done!")
    
        # Metrics
        print("Training accuracy is ",classifier.score(val_d,val_l.ravel()))
        val_predict = classifier.predict(val_d)

        confusion_matrix = sklearn.metrics.confusion_matrix(val_l,val_predict)

        labels_name =["PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou"]

        plot_confusion_matrix(cm = confusion_matrix,labels_name=labels_name,title="RBF SVM 14 classification(Max_second_no1),acc is {}"
     .format(round(classifier.score(val_d,val_l.ravel()),2)), figname="RBF_14_Max_second{}".format(i))


    

    






