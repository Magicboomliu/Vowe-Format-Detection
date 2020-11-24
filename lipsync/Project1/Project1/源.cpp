#include<iostream>
#include<OVRLipSync.h>
#include "Waveform.h"
#include <algorithm>
#include<string>
#include<fstream>

using namespace std;

// 定义最后输出的音节的array， 在头文件中有定义：ovrLipSyncViseme_Count的长度为15
string visemeNames[ovrLipSyncViseme_Count] =
{
	"sil","PP","FF","TH","DD","kk","CH","SS","nn","RR",
	"aa","E","ih","oh","ou",
};

// 获得最大元素的下标
template<unsigned N>
size_t getMaxElementIndex(const float(&array)[N]) {
	auto maxElement = std::max_element(array, array + N);
	return std::distance(array, maxElement);
}
// 打印列表
template<unsigned N>
void printArray(const float(&arr)[N]) {
	std::cout << std::fixed << std::setprecision(2) << arr[0];
	for (unsigned i = 1; i < N; ++i)
		std::cout << "; " << arr[i];
	std::cout << std::endl;
}
// 打印本cpp的用法
void printUsage(const char* progamName) {
	std::cout << "Usage:" << std::endl
		<< "\t" << progamName << " [--print-viseme-distribution] | [--print-viseme-name] [filename.wav]" << std::endl
		<< std::endl
		<< "Read WAV file and print viseme index predictions using the OVRLipSync Enhanced Provider" << std::endl;
}



// This is the main Function
int main(int argc, char** argv)
{
//////////////////////////////////初始化操作////////////////////////////////////////////////////////////
	// 是否打印distribution
	bool printDistribution = argc > 2 && std::string(argv[1]) == "--print-viseme-distribution";
	bool printName = argc > 2 && std::string(argv[1]) == "--print-viseme-name";
	Waveform wav;  // 声明一个wav对象
	ovrLipSyncContext ctx;
	ovrLipSyncFrame frame = {};
	// 初始化全部为0
	float visemes[ovrLipSyncViseme_Count] = { 0.0f };
	frame.visemes = visemes;
	frame.visemesLength = ovrLipSyncViseme_Count;

	// printDistribution = true;
	printName = true;
	ofstream outfile;
	outfile.open("D:\cplus\test.txt",ios::in);

/////////////////////////////文件的I/O操作//////////////////////////////////////////////////////////////
	// 如果没有给参数的话，打印相关的用法
	// Print usage info if invoked without arguments
	if (argc <= 1 || std::string(argv[1]) == "--help") {
		printUsage(argv[0]);
		return 0;
	}

	// 获取文件的信息
	string wavfile_path = argv[1];
	cout << argv[1]<<endl;
	wav = loadWAV(argv[1]);
	cout << "wav文件的Sample Rate是：" << wav.sampleRate << endl;

//////////////////////////////////////// 文件的预处理出操作////////////////////////////////////////////////////////
	// 设置buffer的长度为 10ms * 对应的采样率，等于帧数
	auto bufferSize = static_cast<unsigned int>(wav.sampleRate * 1e-2);
	auto rc = ovrLipSync_Initialize(wav.sampleRate, bufferSize);
	//是否加载成功？
	if (rc != ovrLipSyncSuccess) {
		std::cerr << "Failed to initialize ovrLipSync engine: " << rc << std::endl;
		return -1;
	}
	rc = ovrLipSync_CreateContextEx(&ctx, ovrLipSyncContextProvider_Enhanced, wav.sampleRate, true);
	if (rc != ovrLipSyncSuccess) {
		std::cerr << "Failed to create ovrLipSync context: " << rc << std::endl;
		return -1;
	}
	
	// 开始 for循环
	for (auto offs(0u); offs + bufferSize < wav.sampleCount; offs += bufferSize) {
		// 开始处理每一帧
	     rc = ovrLipSync_ProcessFrame(ctx, wav.floatData() + offs, &frame);
		if (rc != ovrLipSyncSuccess) {
			std::cerr << "Failed to process audio frame: " << rc << std::endl;
			return -1;}
		auto maxViseme = getMaxElementIndex(visemes);
		if (printName) {
			std::cout << visemeNames[maxViseme] << std::endl;
		}
		else {
			std::cout << maxViseme << std::endl;
		}
	}
	rc = ovrLipSync_DestroyContext(ctx);
	rc = ovrLipSync_Shutdown();
	cout << bufferSize << endl;
	system("pause");
	return 0;
}