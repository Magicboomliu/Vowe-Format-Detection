/*******************************************************************************
Filename    :   ProcessWAV.cpp
Content     :   Sample code showing how OVRLipSync API can be used to process PCM data
Created     :   Sep 12th, 2018
Copyright   :   Copyright Facebook Technologies, LLC and its affiliates.
                All rights reserved.

Licensed under the Oculus Audio SDK License Version 3.3 (the "License");
you may not use the Oculus Audio SDK except in compliance with the License,
which is provided at the time of installation or download, or which
otherwise accompanies this software in either electronic or hard copy form.

You may obtain a copy of the License at

https://developer.oculus.com/licenses/audio-3.3/

Unless required by applicable law or agreed to in writing, the Oculus Audio SDK
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*******************************************************************************/

#include "OVRLipSync.h"

#include "Waveform.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

std::string visemeNames[ovrLipSyncViseme_Count] = {
  "sil", "PP", "FF", "TH", "DD",
  "kk", "CH", "SS", "nn", "RR",
  "aa", "E", "ih", "oh", "ou",
};

template<unsigned N>
size_t getMaxElementIndex(const float (&array)[N]) {
    auto maxElement = std::max_element(array, array + N);
    return std::distance(array, maxElement);
}

template<unsigned N>
void printArray(const float (&arr)[N]) {
  std::cout << std::fixed << std::setprecision(2) << arr[0];
  for(unsigned i = 1; i < N; ++i)
    std::cout << "; " << arr[i];
  std::cout << std::endl;
}

void printUsage(const char* progamName) {
  std::cout << "Usage:" << std::endl
    << "\t" << progamName << " [--print-viseme-distribution] | [--print-viseme-name] [filename.wav]" << std::endl
    << std::endl
    << "Read WAV file and print viseme index predictions using the OVRLipSync Enhanced Provider" << std::endl;
}

int main(int argc, const char *argv[]) {
  bool printDistribution = argc > 2 && std::string(argv[1]) == "--print-viseme-distribution";
  bool printName = argc > 2 && std::string(argv[1]) == "--print-viseme-name";

  Waveform wav;

  ovrLipSyncContext ctx;
  ovrLipSyncFrame frame = {};
  float visemes[ovrLipSyncViseme_Count] = {0.0f};

  frame.visemes = visemes;
  frame.visemesLength = ovrLipSyncViseme_Count;

  // Print usage info if invoked without arguments
  if (argc <= 1 || std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return 0;
  }

  try {
    wav = loadWAV(argv[argc - 1]);
  } catch (const std::runtime_error& e) {
    std::cerr << "Failed to load " << argv[1] << " : " << e.what() << std::endl;
    return -1;
  }

  // Feed data to the LipSync engine in 10 ms chunks( I.e. 100 times a second)
  auto bufferSize = static_cast<unsigned int>(wav.sampleRate * 1e-2);

  auto rc = ovrLipSync_Initialize(wav.sampleRate, bufferSize);
  if (rc != ovrLipSyncSuccess) {
    std::cerr << "Failed to initialize ovrLipSync engine: " << rc << std::endl;
    return -1;
  }

  rc = ovrLipSync_CreateContextEx(&ctx, ovrLipSyncContextProvider_Enhanced, wav.sampleRate, true);
  if (rc !=  ovrLipSyncSuccess) {
    std::cerr << "Failed to create ovrLipSync context: " << rc << std::endl;
    return -1;
  }


  for(auto offs(0u); offs + bufferSize < wav.sampleCount; offs += bufferSize) {
    rc = ovrLipSync_ProcessFrame(ctx, wav.floatData() + offs, &frame);
    if (rc != ovrLipSyncSuccess) {
      std::cerr << "Failed to process audio frame: " << rc << std::endl;
      return -1;
    }

    if (printDistribution) {
      printArray(visemes);
      continue;
    }
    auto maxViseme = getMaxElementIndex(visemes);
    if (printName) {
      std::cout << visemeNames[maxViseme] << std::endl;
    } else {
      std::cout << maxViseme << std::endl;
    }
  }
  rc = ovrLipSync_DestroyContext(ctx);
  rc = ovrLipSync_Shutdown();

  return 0;
}
