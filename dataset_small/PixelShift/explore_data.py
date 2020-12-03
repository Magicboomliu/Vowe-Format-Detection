__author__ = "Luke Liu"
#encoding="utf-8"
import  os
import  numpy as np
import librosa
import glob

class PixelShiftSound:
    def __init__(self,sample_rate,frame_duration,frame_shift_duration,datatype):
        ''' 
        datatype = 0 : 10 only
        datatype = 1 : 16 only
        datatype = 2 : 16-8 
        '''
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_shift_duration = frame_shift_duration
        self.datatype =datatype

    def Frame_Normal(self,signal):
        frame_length = int(round(self.frame_duration*self.sample_rate))
        signal_length = len(signal)
        num_frames=int(signal_length//frame_length)
        indices = np.arange(0,frame_length).reshape(1,-1) + np.arange(0,num_frames*frame_length,frame_length).reshape(-1,1)
        frames = signal[indices]
        return num_frames,frames
    def Frame_Shift(self,signal):
        frame_length = int (round(self.frame_duration*self.sample_rate))
        frame_step = int (round(self.frame_shift_duration*self.sample_rate))
        signal_length = len(signal)
        # This ponit need other adjust?
        num_frames = int((signal_length-frame_length)//frame_step) +1
        indices = np.arange(0,frame_length).reshape(1,-1) + np.arange(0,num_frames*frame_length,frame_length).reshape(-1,1)
        nums,length = indices.shape
        frames= np.zeros(indices.shape)
        for i in range(nums):
            data = signal[0+i*frame_step:0+i*frame_step+frame_length]
            frames[i] = data
        return num_frames,frames

    def get_signal_from_file(self,filename):
        signal,sample_rate = librosa.load(filename,sr=self.sample_rate)
        if self.frame_shift_duration==0:
            nums_frames,frames = self.Frame_Normal(signal)
        else:
            nums_frames,frames = self.Frame_Shift(signal)
        return nums_frames,frames
    def get_all_wav_data(self):
        print("Loading the PixelShift Data...")
        wav_list,label_list = self.get_all_wav_files()
        init_f = wav_list[0]
        num_frams,init_frames = self.get_signal_from_file(init_f)
        for index,f in  enumerate(wav_list):
            if index==0:
                continue
            num_frames,frames = self.get_signal_from_file(f)
        
            init_frames = np.vstack((init_frames,frames))
        wav_frame_data = init_frames 
        total_labels=[]
        for index, k in enumerate(label_list):
            with open(k,"r") as f1:
                for line in f1:
                    line = line.strip()
                    string_array = line.split(',')
                    number_array = [float(i) for i in string_array]
                    total_labels.append(number_array)

        total_labels_array = np.zeros((len(total_labels),len(total_labels[0])))
        for index,label in enumerate(total_labels):
            label = np.array(label)
            total_labels_array[index] = label

                
        return wav_frame_data,total_labels_array
        
    def get_all_wav_files(self):
        if self.datatype == 0:
            data_path="10only/S0002/train/S0002"
            label_path="10only/labels"
            wav_file_list=[os.path.join(data_path,f) for f in os.listdir(data_path)]
            label_file_list=[os.path.join(label_path,f) for f in os.listdir(label_path)]
            wav_file_list.sort(key=lambda x: int (x[-7:-4]))
            label_file_list.sort(key=lambda x: int(x[-7:-4]))
            return wav_file_list,label_file_list
        
        if self.datatype == 1:
            data_path="16only/S0002/train/S0002"
            label_path="16only/labels"
            wav_file_list=[os.path.join(data_path,f) for f in os.listdir(data_path)]
            label_file_list=[os.path.join(label_path,f) for f in os.listdir(label_path)]
            wav_file_list.sort(key=lambda x: int (x[-7:-4]))
            label_file_list.sort(key=lambda x: int(x[-7:-4]))
            return wav_file_list,label_file_list
            
        if self.datatype == 2:
            data_path="16_8/S0002/train/S0002"
            label_path="16_8/labels"
            wav_file_list=[os.path.join(data_path,f) for f in os.listdir(data_path)]
            label_file_list=[os.path.join(label_path,f) for f in os.listdir(label_path)]
            wav_file_list.sort(key=lambda x: int (x[-7:-4]))
            label_file_list.sort(key=lambda x: int(x[-7:-4]))
            return wav_file_list,label_file_list
            



# if __name__ == "__main__":
#     '''
#     datatype = 0 : 10 only :frame length is 10ms， No frame shift, sample rate is 16K
#     datatype = 1 : 16 only :frame length is 16ms,  No frame shift, sample rate is 16K 
#     datatype = 2 : 16-8:  frame length is 16ms, frame shift is 8ms. sample rate is 16K 
#     '''
    
#     ps = PixelShiftSound(sample_rate=16000,frame_duration=0.010,frame_shift_duration=0,datatype=0)
#     wav_data,wav_label = ps.get_all_wav_data()
#     print("Wav Frame data：",wav_data.shape)
#     print("Wav Frame label：",wav_label.shape)
    
    
    