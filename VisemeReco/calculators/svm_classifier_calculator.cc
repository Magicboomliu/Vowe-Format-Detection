// Related Libaraies
#include <cstdlib>

#include "absl/container/node_hash_map.h"
#include "absl/strings/str_format.h"
#include "experimental/zihualiu/apps/VisemeReco/calculators/svm_classifier_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "svm.h"
namespace mediapipe {
namespace {
// define input and output variables
constexpr char kMelMfcc[] = "MFCC";
constexpr char kVisemeCategory[] = "VISEME";
}  // namespace

class SvmClassificationCalculator : public CalculatorBase {
 public:
  // constructor functions
  SvmClassificationCalculator() = default;
  // 析够方法
  ~SvmClassificationCalculator() = default;
  // basic functions of calculators
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  // declare a libSVM parameter and a SVM problem
  svm_parameter param;
  svm_model* model;
  int32 classfication_categories_nums;
  int32 classification_type;  // 0 for 5 classification, 1 for 9 classification.
                              // set in Options
};

// register this Calculator
REGISTER_CALCULATOR(SvmClassificationCalculator);

// declare basic functions
::mediapipe::Status SvmClassificationCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<mediapipe::Matrix>();
  cc->Outputs().Index(0).Set<mediapipe::Matrix>();
  // return values
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SvmClassificationCalculator::Open(CalculatorContext* cc) {
  auto options_ =
      cc->Options<experimental::SvmClassificationCalculatorOptions>();
  RET_CHECK(options_.has_classification_type());
  classification_type = options_.classification_type();
  // load the SVM model Here
  if (classification_type == 0) {
    classfication_categories_nums = 5;
    model = svm_load_model(
        "experimental/zihualiu/apps/VisemeReco/training/svm_model/svm5p");
  } else {
    classfication_categories_nums = 9;
    model = svm_load_model(
        "experimental/zihualiu/apps/VisemeReco/training/svm_model/svm9p");
  }

  return ::mediapipe::OkStatus();
}
::mediapipe::Status SvmClassificationCalculator::Process(
    CalculatorContext* cc) {
  mediapipe::Matrix mfcc_matrix =
      cc->Inputs().Index(0).Get<mediapipe::Matrix>();
  // Convert Data into LibSVM formants
  int nums_node = mfcc_matrix.cols();  // cols is 3 : means 3 frames
  int probfeature =
      mfcc_matrix.rows();  // rows is 13 : means 13 dim features of MFCC
  mediapipe::Matrix predict_result(nums_node, classfication_categories_nums);
  if (nums_node > 0 && probfeature > 0) {
    svm_problem prob;
    prob.l = nums_node;
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];

    svm_node* x_space = new svm_node[(probfeature + 1) * nums_node];
    for (int i = 0; i < nums_node; i++) {
      for (int j = 0; j < (probfeature + 1); j++) {
        if (j == probfeature) {
          x_space[(i) * (probfeature + 1) + j].index = -1;
        } else {
          x_space[(i) * (probfeature + 1) + j].index = j + 1;
          x_space[(i) * (probfeature + 1) + j].value = mfcc_matrix.coeff(i, j);
        }
      }
      prob.x[i] = &x_space[i * (probfeature + 1)];
      prob.y[i] = 0;
      double pd[classfication_categories_nums];
      double predict_values = svm_predict_probability(model, prob.x[i], pd);
      for (int k = 0; k < classfication_categories_nums; k++) {
        predict_result(i, k) = pd[k];
      }
    }
    std::unique_ptr<Matrix> output(
        new Matrix(predict_result.rows(), predict_result.cols()));
    *output = predict_result;

    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
    // delete pointer
    delete[] x_space;
    delete[] prob.x;
    delete[] prob.y;
  }

  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe
