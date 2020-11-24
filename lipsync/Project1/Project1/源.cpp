#include<iostream>
#include<OVRLipSync.h>
#include "Waveform.h"
#include <algorithm>
#include<string>
#include<fstream>

using namespace std;

// ���������������ڵ�array�� ��ͷ�ļ����ж��壺ovrLipSyncViseme_Count�ĳ���Ϊ15
string visemeNames[ovrLipSyncViseme_Count] =
{
	"sil","PP","FF","TH","DD","kk","CH","SS","nn","RR",
	"aa","E","ih","oh","ou",
};

// ������Ԫ�ص��±�
template<unsigned N>
size_t getMaxElementIndex(const float(&array)[N]) {
	auto maxElement = std::max_element(array, array + N);
	return std::distance(array, maxElement);
}
// ��ӡ�б�
template<unsigned N>
void printArray(const float(&arr)[N]) {
	std::cout << std::fixed << std::setprecision(2) << arr[0];
	for (unsigned i = 1; i < N; ++i)
		std::cout << "; " << arr[i];
	std::cout << std::endl;
}
// ��ӡ��cpp���÷�
void printUsage(const char* progamName) {
	std::cout << "Usage:" << std::endl
		<< "\t" << progamName << " [--print-viseme-distribution] | [--print-viseme-name] [filename.wav]" << std::endl
		<< std::endl
		<< "Read WAV file and print viseme index predictions using the OVRLipSync Enhanced Provider" << std::endl;
}



// This is the main Function
int main(int argc, char** argv)
{
//////////////////////////////////��ʼ������////////////////////////////////////////////////////////////
	// �Ƿ��ӡdistribution
	bool printDistribution = argc > 2 && std::string(argv[1]) == "--print-viseme-distribution";
	bool printName = argc > 2 && std::string(argv[1]) == "--print-viseme-name";
	Waveform wav;  // ����һ��wav����
	ovrLipSyncContext ctx;
	ovrLipSyncFrame frame = {};
	// ��ʼ��ȫ��Ϊ0
	float visemes[ovrLipSyncViseme_Count] = { 0.0f };
	frame.visemes = visemes;
	frame.visemesLength = ovrLipSyncViseme_Count;

	// printDistribution = true;
	printName = true;
	ofstream outfile;
	outfile.open("D:\cplus\test.txt",ios::in);

/////////////////////////////�ļ���I/O����//////////////////////////////////////////////////////////////
	// ���û�и������Ļ�����ӡ��ص��÷�
	// Print usage info if invoked without arguments
	if (argc <= 1 || std::string(argv[1]) == "--help") {
		printUsage(argv[0]);
		return 0;
	}

	// ��ȡ�ļ�����Ϣ
	string wavfile_path = argv[1];
	cout << argv[1]<<endl;
	wav = loadWAV(argv[1]);
	cout << "wav�ļ���Sample Rate�ǣ�" << wav.sampleRate << endl;

//////////////////////////////////////// �ļ���Ԥ���������////////////////////////////////////////////////////////
	// ����buffer�ĳ���Ϊ 10ms * ��Ӧ�Ĳ����ʣ�����֡��
	auto bufferSize = static_cast<unsigned int>(wav.sampleRate * 1e-2);
	auto rc = ovrLipSync_Initialize(wav.sampleRate, bufferSize);
	//�Ƿ���سɹ���
	if (rc != ovrLipSyncSuccess) {
		std::cerr << "Failed to initialize ovrLipSync engine: " << rc << std::endl;
		return -1;
	}
	rc = ovrLipSync_CreateContextEx(&ctx, ovrLipSyncContextProvider_Enhanced, wav.sampleRate, true);
	if (rc != ovrLipSyncSuccess) {
		std::cerr << "Failed to create ovrLipSync context: " << rc << std::endl;
		return -1;
	}
	
	// ��ʼ forѭ��
	for (auto offs(0u); offs + bufferSize < wav.sampleCount; offs += bufferSize) {
		// ��ʼ����ÿһ֡
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