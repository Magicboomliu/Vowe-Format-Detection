
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


    

if __name__ == "__main__":
    # get the training data
    dct = unpickle("mfcc_t")
    # trainig data and trainig data labels
    trainig_data = dct["MFCC"]
    trainig_data_labels =dct["labels"]
    # data preprocess
    trainig_data = preprocessing.scale(trainig_data)
    print(trainig_data)
    



    # shuffer the training data
    state = np.random.get_state()
    np.random.shuffle(trainig_data)
    np.random.set_state(state)
    np.random.shuffle(trainig_data_labels)
    print(trainig_data.shape)
    print(trainig_data_labels.shape)
    my_training_data = trainig_data[:-200]
    my_training_data_label =trainig_data_labels[:-200]
    print(my_training_data.shape)
    print(my_training_data_label.shape)

    
    # test set
    test_data =trainig_data[-200:]
    test_data_labels =trainig_data_labels[-200:]
    print(test_data.shape)
    print(test_data_labels.shape)
    print(my_training_data_label.ravel())

    accuray_list=[]
    # classifier =svm.SVC(C=2,kernel='rbf',gamma='auto',decision_function_shape='ovr')
    # classifier.fit(my_training_data,my_training_data_label.ravel())
    # print("Training accuracy is ",classifier.score(trainig_data,trainig_data_labels))
    # print("Test accuracy is ",classifier.score(test_data,test_data_labels))
    
    # for i in range(1,7):

    #     classifier =svm.SVC(C=i,kernel='linear',decision_function_shape='ovr')
    #     classifier.fit(my_training_data,my_training_data_label.ravel())
    #     print("Training accuracy is ",classifier.score(trainig_data,trainig_data_labels))
    #     print("Test accuracy is ",classifier.score(test_data,test_data_labels))
    #     accuray_list.append(classifier.score(test_data,test_data_labels))
    #     # get the confussion matrix
    #     test_predict  = classifier.predict(test_data)
    #     confusion_matrix = sklearn.metrics.confusion_matrix(test_data_labels,test_predict)
    #     labels_name =["A","E","I","O","U"]
    #     plot_confusion_matrix(cm=confusion_matrix,labels_name=labels_name,title="SVM-Linear({}) accuracy is {}".format(i,classifier.score(test_data,test_data_labels)),
    #     figname="linear{}".format(i))
        

    #svm with polynomail
    for i in range(1,7,1):
        classifier =svm.SVC(C=2,kernel='poly',gamma='auto',degree=i,decision_function_shape='ovr')
        classifier.fit(my_training_data,my_training_data_label.ravel())
        print("Training accuracy is ",classifier.score(trainig_data,trainig_data_labels))
        print("Test accuracy is ",classifier.score(test_data,test_data_labels))
        accuray_list.append(classifier.score(test_data,test_data_labels))
        # get the confussion matrix
        test_predict  = classifier.predict(test_data)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_data_labels,test_predict)
        labels_name =["A","E","I","O","U"]
        plot_confusion_matrix(cm=confusion_matrix,labels_name=labels_name,title="SVM-poly({}) accuracy is {}".format(i,classifier.score(test_data,test_data_labels)),
        figname="Poly{}".format(i))
    

    
    plt.plot(range(1,7,1),accuray_list)
    for i in range(1,7,1):
        plt.Circle((i,accuray_list[i-1]),radius=10,color='red')
    plt.xlabel("Ploy's Degree")
    plt.ylabel("Accuracy rate on Test data")
    plt.title("Ploy SVM with different C")
    plt.savefig("Ploy_accuracy",format='png')









    # # start svm classification
    # classifier = svm.SVC(C=2,kernel='poly',gamma='auto',degree=6,decision_function_shape='ovr')
    # classifier.fit(my_training_data,my_training_data_label.ravel())
    
    # # get accuracy rate
    # print("Training accuracy is ",classifier.score(trainig_data,trainig_data_labels))
    # print("Test accuracy is ",classifier.score(test_data,test_data_labels))
    
    # # get the confussion matrix
    # test_predict  = classifier.predict(test_data)
    # confusion_matrix = sklearn.metrics.confusion_matrix(test_data_labels,test_predict)
    # labels_name =["A","E","I","O","U"]
    # plot_confusion_matrix(cm=confusion_matrix,labels_name=labels_name,title="SVM-poly(6) accuracy is {}".format(classifier.score(test_data,test_data_labels)),
    # figname="ploy6")
    


