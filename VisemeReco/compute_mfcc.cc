#include <cstdlib>
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/audio_decoder.h"
#include "mediapipe/util/audio_decoder.pb.h"
#include "pixelshift/util/one_euro_filter.h"
#include "raylib.h"
#include "rlgl.h"


DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");  // Graph的地址
DEFINE_string(input_audio_path, "", "Full path of audio to load. ");    // Input_audio的地址

// 定义输入的SidePackage 以及 输入输出流
constexpr char kInputAudioSidePacket[] = "input_audio";
constexpr char kOutputMfcc[] = "mel_mfcc";
constexpr char kInputStream[] = "mono_waveform";

mediapipe::Matrix mfcc_matrix;


::mediapipe::Status RunMPPGraph() {
  // 设置Audio decoder的一些属性
  auto audio_decoder_options =
      absl::make_unique<mediapipe::AudioDecoderOptions>();
  auto audio_stream = audio_decoder_options->add_audio_stream();
  audio_stream->set_stream_index(0);      //保留单声道 ？
  audio_stream->set_allow_missing(false);
  audio_stream->set_ignore_decode_failures(false);
  audio_stream->set_output_regressing_timestamps(false);

  LOG(INFO) << "完成 audio decoder 设置";

  // 声明一个audio decoder
  auto decoder = absl::make_unique<mediapipe::AudioDecoder>();
  //将设置的属性 赋予刚刚声明的 audio decoder
  MP_RETURN_IF_ERROR(
      decoder->Initialize(FLAGS_input_audio_path, *audio_decoder_options));

  // 声明一个Time Header
  auto header = absl::make_unique<mediapipe::TimeSeriesHeader>();
    if (decoder
          ->FillAudioHeader(audio_decoder_options->audio_stream(0),
                            header.get())
          .ok()) {
    LOG(INFO) << "设置Header成功";
  }
  LOG(INFO) << "完成audio decoder的初始化";

  // 设置声音为单声道：Mono
  header->set_num_channels(1);
  // 验证 Header中是否有东西
  LOG(INFO) << "Header: " << header->DebugString();

///////////////////////////////////////////////////////////////////////
  // Setup graph

// 加载图的内容，并且赋值给calculator_graph_config_contents 这个字符串变量
std::string calculator_graph_config_contents;
 MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
 
 //加载图中需要设置的Config
 mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

auto graph = absl::make_unique<mediapipe::CalculatorGraph>();

MP_RETURN_IF_ERROR(graph->Initialize(config));

LOG(INFO)<<"建图成功"<<"\n";

// 建立图的输出
graph->ObserveOutputStream(kOutputMfcc,
                             [&](const mediapipe::Packet& mfcc_packet) {
                                mfcc_matrix = mfcc_packet.Get<mediapipe::Matrix>();
                               return mediapipe::OkStatus();
                             });

graph->StartRun({}, {{kInputStream, mediapipe::Adopt(header.release())}});
int options_index = -1;

// Start Processing The Audio
while (true){

mediapipe::Packet audio_packet;

//设置退出while loop的条件
if (!decoder->GetData(&options_index, &audio_packet).ok()) {
      break;
    }
// 得到音频数据的data
std::unique_ptr<mediapipe::Matrix> mono_wave(
    new mediapipe::Matrix(audio_packet.Get<mediapipe::Matrix>().row(0)));

// 把音频数据package和InputStream进行binding
MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
    kInputStream,
    mediapipe::Adopt(mono_wave.release()).At(audio_packet.Timestamp())));

// 获取音频的时间戳
    int64 audio_frame_timestamp;
    audio_frame_timestamp = audio_packet.Timestamp().Microseconds();
   
LOG(INFO)<<"这个是 mfcc_matrix 输出cols:"<<mfcc_matrix.cols()<<"\n";
LOG(INFO)<<"这个是 mfcc_matrix 输出rows:"<<mfcc_matrix.rows()<<"\n";
}

// while循环执行结束
LOG(INFO) << "Shutting down.";
MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
MP_RETURN_IF_ERROR(graph->WaitUntilDone());
return mediapipe::OkStatus();

}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ::mediapipe::Status run_status = RunMPPGraph();

  // ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
