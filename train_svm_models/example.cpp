#include <iostream>
#include <string>
#include <vector>
#include "svm.h"

using namespace std;
svm_parameter param;

void init_param() {
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

int main(int argc, char const* argv[]) {
  /* code */
  init_param();
  cout << "Trying SVM to solve the problem" << endl;
  svm_problem prob;             // Svm_problem 儲蓄所有參加計算的樣本
  prob.l = 4;                   //样本数
  prob.y = new double[prob.l];  //指向樣本類別的標籤數組
  double d;
  int probfeature = 2;  //样本特征维数
  if (param.gamma == 0) param.gamma = 0.5;
  // svm_node 儲蓄單一向量的單個特徵
  // 生成一個svm_node的array, 之所以要加1，是因爲最後一個特徵爲NULL, index=-1
  svm_node* x_space =
      new svm_node[(probfeature + 1) *
                   prob.l];  //样本特征存储空间 features * NUM_of_Samples

  prob.x = new svm_node*[prob.l];  //每一个X指向一个样本

  cout << "size: " << sizeof(x_space)
       << endl;  // size是8的原因是由於 SVM_node 中每一個特徵有2個features

  // 初始化數據，一共4個，每個數據包含2個dim, 身高和體重
  // 第一個數據
  x_space[0].index = 1;
  x_space[0].value = 190;
  x_space[1].index = 2;
  x_space[1].value = 70;
  x_space[2].index = -1;
  // 把第一個數據給prob.x
  prob.x[0] = &x_space[0];
  prob.y[0] = 1;
  // 第二個數據
  x_space[3].index = 1;
  x_space[3].value = 180;
  x_space[4].index = 2;
  x_space[4].value = 80;
  x_space[5].index = -1;
  prob.x[1] = &x_space[3];
  prob.y[1] = 1;
  // 第三個數據
  x_space[6].index = 1;
  x_space[6].value = 161;
  x_space[7].index = 2;
  x_space[7].value = 45;
  x_space[8].index = -1;
  prob.x[2] = &x_space[6];
  prob.y[2] = -1;
  // 第四個數據
  x_space[9].index = 1;
  x_space[9].value = 163;
  x_space[10].index = 2;
  x_space[10].value = 47;
  x_space[11].index = -1;
  prob.x[3] = &x_space[9];
  prob.y[3] = -1;
  // training the svm model
  svm_model* model = svm_train(&prob, &param);
  // 保存訓練的數據
  svm_save_model("/home/liuzihua/lzh_intern/MyModel", model);

  // 加載模型
  // svm_model *model_loaded = load_model("/home/liuzihua/lzh_intern/MyModel");

  cout << "SVM Training is Done" << endl;
  // Come to predict

  // predict 身高180cm, 体重85kg
  svm_node xnode[3];
  xnode[0].index = 1;
  xnode[0].value = 161;
  xnode[1].index = 2;
  xnode[1].value = 85;
  xnode[2].index = -1;
  d = svm_predict(model, xnode);
  cout << "Print Prediction is :" << d << endl;

  return 0;
}
