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

cmd=${1:-specturm}
shift

$cmd "$@"
