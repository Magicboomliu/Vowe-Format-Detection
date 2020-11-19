#include<iostream>
#include <string>
using namespace std;
#include "svm.h"
#include <fstream> // 引入文件流函數頭
#include <vector>

// 有關音頻數據庫的相關細節

// 标准音频的采样率为16khz
// 每个标准音频帧（AudioFrame）的长度应该为16ms,Step为8ms，进行标注（即判断这16ms内音节所属的元音，标注在起始）。
// 具体操作为每隔8ms建立一个label,此label的内容是判断从这这个时间点开始后面16ms的音频数据的具体归属。
// 标记的标签为 A、E、I、O、U、None(No sound)、Other( sound except for a、e、i、o、u) 一共7种label。


// STEPS TO TRIAN A SVM MODEL:
// (1) 讀取到所有數據集中的數據，並且從中提取出來mfcc特徵的數據，存入本地文件。
// (2) 讀取本地文件，將數據轉化爲指定的libSVM的格式
// (3) 確定Prob的個數，並且爲prob賦值
// (4) 開始訓練， 查看訓練的結果。 最後保存svm模型。

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 定義一個函數讀取本地文件，將數據轉化爲指定的float vector的格式
 std::vector<std::vector<double>> readMfcc(string mfcc_path){
    ifstream infile;
    ofstream outfile;
    infile.open(mfcc_path,ios::in);
    // 判斷文件讀取操作是否成功
    if(!infile.is_open())
    {
    cout<<"Open File Operation is failed!"<<endl;
    }
    // 此文件的mfcc特徵
    std::vector<std::vector<double>> mfcc;

    std::vector<double> mfcc_features;
    int values_count = 0;    
    while(! infile.eof()){
        string data;
        string true_mfcc_string;
        double mfcc_value;
        // 每次讀取一次數據，count++
        infile>>data;
        
        if (data[data.length()-1] == ','){
            true_mfcc_string = data.substr(0,data.length()-1); }
        else {
            true_mfcc_string =data;}
        mfcc_value = atof(true_mfcc_string.c_str());
        values_count++;
        mfcc_features.push_back(mfcc_value);
        if (values_count==13){
            //清空加初始化操作
                mfcc.push_back(mfcc_features);
                values_count=0;
                mfcc_features.clear();}
                          }
                infile.close();
            return mfcc;
}

// 將數據從 double vector 類型轉化爲 libSVM可識別的 svm_node的格式
svm_node* changeToLibSVm(std::vector<double> dv){  // 確定每個數據有 13 個維度
   int profeature = 13;
   //創建一個指針變量
   svm_node* x_space = new svm_node[profeature+1];
   for(int i=0;i<dv.size();i++){
        x_space[i].index = i+1;
        x_space[i].value = dv[i];
   }
       x_space[dv.size()].index = -1;
       
       return x_space;
}

// Get 每個元音的svm_node信息，保存在 std::vector<svm_node*> 裏面
std::vector<svm_node*> getAllSvmNode(std::vector<std::vector<double>> dvv){
   // 首先提取出來每個vector進行svm_node的轉換
   std::vector<svm_node*> svm_ptr_vector;
   for (int i=0;i<dvv.size();i++){
       std::vector<double> dv = dvv[i];
       
       // get 到了一個特徵13維度的
       svm_node* x_space = changeToLibSVm(dv);
       svm_ptr_vector.push_back(x_space);
   }
   //返回這個vec
   return svm_ptr_vector;
}

// Init SVM parameters
void init_param(svm_parameter &param) {
  param.svm_type = C_SVC;
  param.kernel_type = RBF;
  param.degree = 3;
  param.gamma = 0.0001;
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 10;
  param.eps = 1e-5;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
}


// 把他賦值給prob
void give_it_to_prob(svm_problem* prob,std::vector<svm_node*> mfcc_,int label,int begin_index)
{   // mfccs_svm_nodes 的size 的大小就是 每個元音有多少個 樣本
     for(int i =0;i<mfcc_.size();i++){
        svm_node* x_space = mfcc_[i];
        prob->x[i+begin_index] = &x_space[0];
        prob->y[i+begin_index] = label;   
     }
     
}

/// 說明： mfccs_svm_nodes 的size 的大小就是 每個元音有多少個 樣本
///        x_space的大小爲14，就是一個vector需要多大的 dim

int main(int argc, char const *argv[])

 {   
    // MFCC FILES : A、E、I、O、U
    string MFCC_path = "/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_a";
    string MFCC_A_path ="/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_a";
    string MFCC_E_path ="/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_e";
    string MFCC_I_path ="/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_i";
    string MFCC_O_path ="/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_o";
    string MFCC_U_path ="/home/liuzihua/lzh_intern/index_files/train_svm_models/mfcc_string_data/mfcc_u";
    string MFCC_Path_List[] ={ MFCC_A_path,MFCC_E_path,MFCC_I_path,MFCC_O_path,MFCC_U_path };
    std::vector<std::vector<double>>  mfccs;
    std::vector<svm_node*> mfccs_svm_nodes;
    std::vector<std::vector<svm_node*>> mfccs_svm_nodes_total;
    // Init SVM parameters
    svm_parameter param;
    init_param(param);
    svm_problem prob;
    
    svm_model *model;

    
    //////////////////////////////MFCC文件讀取操作/////////////////////////////////////
    int total_mfcc_node_nums = 0;
    for (int i=0;i<(sizeof(MFCC_Path_List)/sizeof(MFCC_Path_List[0]));i++){
        mfccs = readMfcc(MFCC_Path_List[i]);
        
        cout<<"This is MFCC DATA SIZE : "<<mfccs.size()<<endl;
        // 得到每一個元音的libsvm數據
        mfccs_svm_nodes = getAllSvmNode(mfccs);
        mfccs_svm_nodes_total.push_back(mfccs_svm_nodes);
        cout<<"This is the MFCC SVM NODE SIZE : "<<mfccs_svm_nodes.size()<<endl;
        // 一共有多少 ss vm_node instances
        total_mfcc_node_nums+=mfccs_svm_nodes.size();
        cout<<"Successfully Read the "<<i+1<<"個mfcc文件"<<endl;
   }
    cout<<"MFCC文件讀取操作"<<endl;

    // 定義一個 svm問題， 初始化此問題。
    prob.l = total_mfcc_node_nums; // 定義這個參與進行svm訓練的樣本的個數。
    prob.y = new double[prob.l];// 定義這些樣本的標籤
    prob.x = new svm_node* [prob.l]; //定義x的內存
    int current = 0;
    for (int i = 0;i<mfccs_svm_nodes_total.size();i++){
      std::vector<svm_node*> mfccs_twice;
      mfccs_twice =mfccs_svm_nodes_total[i];
      give_it_to_prob(&prob,mfccs_twice,i,current);
      current+=mfccs_twice.size();
    }
    
    cout<<"SVM-Problem 初始化成功"<<endl;
    ///////////////////////////開始訓練，保存訓練的結果////////////////////////////////////////////
    model = svm_train(&prob,&param);
    svm_save_model("/home/liuzihua/lzh_intern/Vowel_classification", model);

    cout<<"svm模型訓練成功"<<endl;
    
    delete[] prob.x;
    delete[] prob.y;

    

    /* code */
    return 0;
}
