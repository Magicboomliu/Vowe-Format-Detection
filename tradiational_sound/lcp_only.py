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

from sklearn import svm
from sklearn import preprocessing
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

# get confusion matrix
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

def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

def Gussaine2d(m1,s1,m2,s2,x1,x2):

    return np.exp(-0.5 *(math.pow((x1-m1),2)/math.pow(s1,2) +math.pow((x2-s2),2)/math.pow(s2,2)))

def IsPossible(x,mean,std):
    return (x>mean-3*std) and (x<mean+3*std)
   

 ### F1 and F2 threshold for each formant  
    #   //////// formant to vowels /////////
    #   // 	a        e  	  I	     o	    u
    #   //                F1
    #   // 873.5	 561.5	 295	  593	   348
    #   // 67.9	   49.5    20.8   50.6	 28.7
    #   //                F2
    #   // 1269.5	1248.5	2555.5	882	  731.5
    #   // 120.6	 69.4	  102.7   59.9	55.5






if __name__ == "__main__":
    n_vowe_nums=5
    data = unpickle("lcp_t")
    mydata = data["LCPs"]
    mydata_labels =data["labels"]
    print(mydata.shape)
    print(mydata_labels.shape)
    
    # data from papers
    f1_mean_list=[873.5,561.5,295,593,348]
    f1_std_list=[67.9,49.5,20.8,50.6,28.7]
    f2_mean_list=[1269.5,1248.5,2555.5,882,731.5]
    f2_std_list=[120.6,69.4,102.7,59.9,55.5]

   
    
    predict_labels =[]
    detected_labels =[]
    detected_ground_true=[]
    for ee,frame in enumerate(mydata):
        # Get F1 and F1 value
        f1 =frame[0]
        f2 =frame[1]
        possible= False
        #         for (int i = 0; i < vowel_count_; i++)
        #   possible += IsPossible(f1, f1_mean[i], f1_std[i]) &&
        #               IsPossible(f2, f2_mean[i], f2_std[i]);
        for i in range(5):
            possible+= (IsPossible(f1,f1_mean_list[i],f1_std_list[i]) and IsPossible(f2,f2_mean_list[i],f2_std_list[i]))
      
        if possible:
            p_a =Gussaine2d(m1=f1_mean_list[0],s1=f1_std_list[0],m2=f2_mean_list[0],s2=f2_std_list[0],x1=f1,x2=f2)
            p_e =Gussaine2d(m1=f1_mean_list[1],s1=f1_std_list[1],m2=f2_mean_list[1],s2=f2_std_list[1],x1=f1,x2=f2)
            p_i =Gussaine2d(m1=f1_mean_list[2],s1=f1_std_list[2],m2=f2_mean_list[2],s2=f2_std_list[2],x1=f1,x2=f2)
            p_o =Gussaine2d(m1=f1_mean_list[3],s1=f1_std_list[3],m2=f2_mean_list[3],s2=f2_std_list[3],x1=f1,x2=f2)
            p_u =Gussaine2d(m1=f1_mean_list[4],s1=f1_std_list[4],m2=f2_mean_list[4],s2=f2_std_list[4],x1=f1,x2=f2)
            proba_list=[p_a,p_e,p_i,p_o,p_u]
            max_id=0
            for ii,pro in enumerate(proba_list):
                if pro>proba_list[max_id]:
                    max_id =ii
                
            output = max_id

            detected_labels.append(output)
            detected_ground_true.append(mydata_labels[ee])
        # else:
        #     output=666
        
        # predict_labels.append(output)
    ## total sumS
    
    predict_labels = np.array(detected_labels)
    print(predict_labels.shape)
    accuaracy =sklearn.metrics.accuracy_score(detected_ground_true,detected_labels)
    confusion_matrix =sklearn.metrics.confusion_matrix(detected_ground_true,detected_labels)
    labels_name =["A","E","I","O","U"]
    plot_confusion_matrix(cm=confusion_matrix,labels_name=labels_name,title="LPC(only)- acc is :{}".format(accuaracy),figname="LCP_ONLY")


    print(accuaracy)
  
        




