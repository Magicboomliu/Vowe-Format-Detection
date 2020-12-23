function simple() {
cd ../../../..
GLOG_alsologtostderr=1 bazel run experimental/zihualiu/apps/VisemeReco/simple --define MEDIAPIPE_DISABLE_GPU=1 \
 -- \
 --calculator_graph_config_file experimental/zihualiu/apps/VisemeReco/graphs/simple.pbtxt \
 --input_audio_path experimental/zihualiu/apps/VisemeReco/resources/haoyuan.mp3 
}

function specturm(){
cd ../../../..
GLOG_alsologtostderr=1 bazel run experimental/zihualiu/apps/VisemeReco/specturm --define MEDIAPIPE_DISABLE_GPU=1 \
 -- \
 --calculator_graph_config_file experimental/zihualiu/apps/VisemeReco/graphs/specturm.pbtxt \
 --input_audio_path experimental/zihualiu/apps/VisemeReco/resources/haoyuan.mp3 
}

function mfcc(){
cd ../../../..
GLOG_alsologtostderr=1 bazel run experimental/zihualiu/apps/VisemeReco/mfcc --define MEDIAPIPE_DISABLE_GPU=1 \
 -- \
 --calculator_graph_config_file experimental/zihualiu/apps/VisemeReco/graphs/mel_mfcc.pbtxt \
 --input_audio_path experimental/zihualiu/apps/VisemeReco/resources/haoyuan.mp3 
}

function full_simple(){
cd ../../../..
GLOG_alsologtostderr=1 bazel run experimental/zihualiu/apps/VisemeReco/full_simple --define MEDIAPIPE_DISABLE_GPU=1 \
 -- \
 --calculator_graph_config_file experimental/zihualiu/apps/VisemeReco/graphs/full_simple.pbtxt \
 --input_audio_path experimental/zihualiu/apps/VisemeReco/resources/haoyuan.mp3 
}

function full_mlp(){
cd ../../../..
GLOG_alsologtostderr=1 bazel run experimental/zihualiu/apps/VisemeReco/full_mlp --define MEDIAPIPE_DISABLE_GPU=1 \
 -- \
 --calculator_graph_config_file experimental/zihualiu/apps/VisemeReco/graphs/full_mlp.pbtxt \
 --input_audio_path experimental/zihualiu/apps/VisemeReco/resources/haoyuan.mp3 

}

cmd=${1:-full_mlp}
shift

$cmd "$@"
