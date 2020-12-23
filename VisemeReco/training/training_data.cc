#include <iostream>
using namespace std;
#include<fstream>
#include<vector>
#include "experimental/zihualiu/apps/VisemeReco/training/mfcc5.pb.h"
#include "svm.h"

// 从protobuf中读取数据
void readData_proto(const char* proto_path,vector<vector<double>> &np_data){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    tutorial::TwoDArray twoDArray;
    fstream input(proto_path, ios::in | ios::binary);
    if (!twoDArray.ParseFromIstream(&input)) {
      cout << "Failed to parse numpy array." << endl;
      return;
    }
    // 读取protobuf的数据
    for (int i = 0; i < twoDArray.array_size(); i++) {
        tutorial::OneDArray node_array = twoDArray.array(i);
        vector<double> data_vector;
        for (int j=0;j<node_array.value_size();j++){
            data_vector.push_back(node_array.value(j));
        }
        np_data.push_back(data_vector);
    }
    input.close();
}
void readlabel_proto(const char* proto_path,vector<double>&np_data){
   GOOGLE_PROTOBUF_VERIFY_VERSION;
   tutorial::OneDArray oneDArray;
   fstream input(proto_path, ios::in | ios::binary);
   if (!oneDArray.ParseFromIstream(&input)) {
      cout << "Failed to parse numpy array." << endl;
      return;
    }
    for (int i=0;i<oneDArray.value_size();i++){
        np_data.push_back(oneDArray.value(i));
    }
   input.close();

}
// 定义一个SVM param
svm_parameter param;
void init_svm(){
    param.svm_type = C_SVC;
    param.degree =5;
    param.kernel_type = RBF;
    param.gamma =0.1;
    param.coef0 =0;
    param.nu =0.5;
    param.cache_size=100;
    param.C=10;
    param.eps = 1e-5;
    param.shrinking =1;
    param.nr_weight=0;
    param.probability =0;
    param.weight_label = NULL;
    param.weight = NULL;
}
 // 把数据转换为libSVM node的格式
void libsvm_formant(svm_problem &prob,vector<vector<double>>data, vector<double>label){
    // 记录一共有多少样本，以及样本的维度
    const int probnums = data.size();
    const int probfeatures = data[0].size();

    prob.l = probnums; // 样本个数
    prob.y = new double [prob.l]; // 初始化样本标签
    svm_node *x_space = new svm_node[(probfeatures+1)*prob.l];
    prob.x = new svm_node *[prob.l];
    for (int i=0;i<prob.l;i++){
      for(int j=0;j<(probfeatures+1);j++){
        if(j==probfeatures){
                x_space[i*(probfeatures+1)+j].index =-1;
            }
            else{
                x_space[i*(probfeatures+1)+j].index = j+1;
                x_space[i*(probfeatures+1)+j].value = data[i][j];
            }
        }
        prob.x[i] = &x_space[i*(probfeatures+1)];
        prob.y[i] = label[i];
    }
  

}
// 得到训练的正确率
double acc_rate(svm_model *model,vector<vector<double>>data,vector<double>label){
    svm_problem prob;
    libsvm_formant(prob,data,label);
    int acc =0;
    for (int i=0;i<prob.l;i++){
        double p = svm_predict(model,prob.x[i]);
        if (p==label[i]){acc=acc+1;}
    }
    return acc*1.0/prob.l;
}
// 训练模型
void training_svm(int type, const char* absolute_saved_path){
    // 从protobuf中读取数据
    vector<vector<double>> mfcc5_data;
    vector<double>mfcc5_label;
    if (type==0){
    readData_proto("experimental/zihualiu/apps/VisemeReco/training/data/mfcc5_data_proto",mfcc5_data);
    readlabel_proto("experimental/zihualiu/apps/VisemeReco/training/data/mfcc5_label_proto",mfcc5_label);
    }
    else if (type==1){
    readData_proto("experimental/zihualiu/apps/VisemeReco/training/data/mfcc9_data_proto",mfcc5_data);
    readlabel_proto("experimental/zihualiu/apps/VisemeReco/training/data/mfcc9_label_proto",mfcc5_label);
    }
    else{
     cout<<"type is only support '0' and '1' , '0' is 5 category, '1' is 9 category"<<endl;
     return;
    }
    
    //以8:2划分训练集和测试集
    const int prob_nums = mfcc5_data.size();
    const int prob_training_nums = int(prob_nums*0.8);
    const int prob_testing_nums = int(prob_nums*0.2);

    vector<vector<double>> training_data;
    vector<double> training_label;
    training_data = vector<vector<double>>(mfcc5_data.begin(), mfcc5_data.begin()+prob_training_nums);
    training_label = vector<double>(mfcc5_label.begin(),mfcc5_label.begin()+prob_training_nums);

    vector<vector<double>> testing_data;
    vector<double> testing_label;
    testing_data = vector<vector<double>>(mfcc5_data.begin()+prob_training_nums, mfcc5_data.begin()+prob_training_nums+prob_testing_nums);
    testing_label = vector<double>(mfcc5_label.begin()+prob_training_nums,mfcc5_label.begin()+prob_training_nums+prob_testing_nums);
   
    // 初始化SVM问题和参数
    init_svm(); // 初始化SVM参数
    svm_problem prob; // 定义一个SVM问题


    // 训练部分
    libsvm_formant(prob,training_data,training_label);
    cout<<"Begin Training,Please Wait"<<endl;

    svm_model *model = svm_train(&prob,&param);
    
    cout<<"Training is done"<<endl;

    // 评估部分
    double acc_training = acc_rate(model,training_data,training_label);
    double acc_testing = acc_rate(model,testing_data,testing_label);
    cout<<"Accuracy rate on Training is : "<<acc_training *100<<"%"<<endl;
    cout<<"Accuracy rate on Testing is : "<<acc_testing *100<<"%"<<endl;

    // 保存模型
    svm_save_model(saved_path,model);
    cout<<endl;
    cout<<"categories of classification is  :  "<<model->nr_class<<endl;
}

int main(int argc, char const *argv[])
{   
    training_svm(1,"/home/liuzihua/Desktop/media/newest/mediapipe/experimental/zihualiu/apps/VisemeReco/training/svm9");
    return 0;
}
