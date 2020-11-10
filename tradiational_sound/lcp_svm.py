import pickle
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    data = unpickle("lcp_t")
    mydata = data["LCPs"]
    mydata_labels =data["labels"]
    print(mydata.shape)
    print(mydata_labels.shape)
    # data preprocess, centerlization
    mydata = preprocessing.scale(mydata)
    print(mydata.shape)

    #  training set and testing set spilt

    # shuffer the training data
    state = np.random.get_state()
    np.random.shuffle(mydata)
    np.random.set_state(state)
    np.random.shuffle(mydata_labels)
    my_training_data = mydata[:-200]
    my_training_data_label =mydata_labels[:-200]
    print("training data size:",my_training_data.shape)
    print("training label size:",my_training_data_label.shape)    
    # test set
    test_data =mydata[-200:]
    test_data_labels =mydata_labels[-200:]
    print("testing data size:",test_data.shape)
    print("testing label size:",test_data_labels.shape)

    # 
    accuray_list=[]
    for i in range(1,7):
        classifier =svm.SVC(C=2,kernel='poly',gamma='auto',degree=i,decision_function_shape='ovr')
        classifier.fit(my_training_data,my_training_data_label.ravel())
        print("Training accuracy is ",classifier.score(my_training_data,my_training_data_label))
        print("Test accuracy is ",classifier.score(test_data,test_data_labels))
        accuray_list.append(classifier.score(test_data,test_data_labels))
        # get the confussion matrix
        test_predict  = classifier.predict(test_data)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_data_labels,test_predict)
        labels_name =["A","E","I","O","U"]
        plot_confusion_matrix(cm=confusion_matrix,labels_name=labels_name,title="Poly-linear_LPC({}) accuracy is {}".format(i,classifier.score(test_data,test_data_labels)),
        figname="Poly_LPC{}".format(i))
    
    plt.plot(range(1,7),accuray_list)
    plt.xlabel("Poly's Degree")
    plt.ylabel("Accuracy rate on Test data")
    plt.title("Poly SVM with different C")
    plt.savefig("LCP_accuracy_Poly",format='png')


        








    

