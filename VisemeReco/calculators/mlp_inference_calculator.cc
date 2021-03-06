// Related Libaraies
#include <cstdlib>
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace {
// define input and output variables
constexpr char kMelMfcc[] = "MFCC";
constexpr char kVisemeCategory[] = "VISEME";
}  // namespace

class MlpInferenceCalculator : public CalculatorBase {
 public:
  // 构造方法
  MlpInferenceCalculator() = default;
  // 析够方法
  ~MlpInferenceCalculator() = default;
  // 声明基本的函数类型
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};

// 注册此Calculator
REGISTER_CALCULATOR(MlpInferenceCalculator);

// 开始实现类内声明的函数
::mediapipe::Status MlpInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<mediapipe::Matrix>();
  cc->Outputs().Index(0).Set<mediapipe::Matrix>();
  // return values
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MlpInferenceCalculator::Open(CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}
::mediapipe::Status MlpInferenceCalculator::Process(CalculatorContext* cc) {
  mediapipe::Matrix mfcc_matrix =
      cc->Inputs().Index(0).Get<mediapipe::Matrix>();

  std::unique_ptr<Matrix> output(
      new Matrix(mfcc_matrix.rows(), mfcc_matrix.cols()));
  *output = mfcc_matrix;

  cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe
