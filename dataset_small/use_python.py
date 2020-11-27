__author__ = "Luke Liu"
#encoding="utf-8"
import  os
import  numpy as np
import librosa
import glob

from explore_data import PixelShiftSound

if __name__ == "__main__":
    '''
    datatype = 0 : 10 only :frame length is 10ms， No frame shift, sample rate is 16K
    datatype = 1 : 16 only :frame length is 16ms,  No frame shift, sample rate is 16K 
    datatype = 2 : 16-8:  frame length is 16ms, frame shift is 8ms. sample rate is 16K 
    
    '''
    ps = PixelShiftSound(sample_rate=16000,frame_duration=0.010,frame_shift_duration=0,datatype=0)
    wav_data,wav_label = ps.get_all_wav_data()
    print("Wav Frame data：",wav_data.shape)
    print("Wav Frame label：",wav_label.shape)


    
    
    