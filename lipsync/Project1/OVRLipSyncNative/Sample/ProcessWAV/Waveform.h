/*******************************************************************************
Filename    :   Waveform.h
Content     :   Primitive PCM file reader
Created     :   Mar 25th, 2018
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

#pragma once

#include <fstream>
#include <sstream>
#include <memory>
#include <cstdint>

struct Waveform {
  Waveform(unsigned numSamples = 0, unsigned rate = 9600)
      : sampleRate(rate), sampleCount(numSamples) {}
  Waveform(const Waveform&) = delete;
  Waveform(Waveform&& other)
      : sampleRate(other.sampleRate),
        sampleCount(other.sampleCount),
        floats_(std::move(other.floats_)),
        shorts_(std::move(other.shorts_)) {
    other.sampleCount = 0;
  }

  Waveform& operator=(Waveform&& other) {
    sampleCount = other.sampleCount;
    sampleRate = other.sampleRate;
    floats_ = std::move(other.floats_);
    shorts_ = std::move(other.shorts_);
    other.sampleCount = 0;
    return *this;
  }

  unsigned sampleRate;
  unsigned sampleCount;
  float* floatData() {
    if (!sampleCount || floats_) {
      return floats_.get();
    }
    floats_.reset(new float[sampleCount]);
    if (shorts_) {
      for (unsigned cnt = 0; cnt < sampleCount; ++cnt)
        floats_[cnt] = shorts_[cnt] / 32767.0f;
    }
    return floats_.get();
  }

  int16_t* shortData() {
    if (!sampleCount || shorts_) {
      return shorts_.get();
    }
    shorts_.reset(new int16_t[sampleCount]);
    if (floats_) {
      for (unsigned cnt = 0; cnt < sampleCount; ++cnt)
        shorts_[cnt] = int16_t(floats_[cnt] * 32767);
    }
    return shorts_.get();
  }

 private:
  std::unique_ptr<float[]> floats_;
  std::unique_ptr<int16_t[]> shorts_;
};

#pragma pack(push, 1)
struct wavChunkHeader {
  char chunkID[4];
  uint32_t chunkSize;
};
struct wavFormatChunk {
  wavChunkHeader chunkHeader;
  uint16_t format;
  uint16_t chanNo;
  uint32_t samplesPerSec;
  uint32_t bytesPerSec;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
};

struct wavHeader {
  wavChunkHeader masterChunk;
  char WAVEmagic[4];
  wavFormatChunk format;
  char padding[64];
};
#pragma pack(pop)

bool isChunkIDEqual(const char* ID1, const char* ID2) {
  return *reinterpret_cast<const uint32_t*>(ID1) == *reinterpret_cast<const uint32_t*>(ID2);
}

Waveform loadWAV(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);

  if (!ifs.is_open()) {
    throw std::runtime_error("Can't open " + filename);
  }
  wavHeader hdr;
  if (!ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
    throw std::runtime_error("Can not read WAV header");
  // Check master chunk IDs
  if (!isChunkIDEqual(hdr.masterChunk.chunkID, "RIFF") || !isChunkIDEqual(hdr.WAVEmagic, "WAVE"))
    throw std::runtime_error("Corrupted RIFF header");
  // Validate format chunk
  if (!isChunkIDEqual(hdr.format.chunkHeader.chunkID, "fmt ") || hdr.format.chunkHeader.chunkSize < 16)
    throw std::runtime_error("Corrupted runtime header");
  if (hdr.format.format != 1)
    throw std::runtime_error("Only PCM format is supported");
  if (hdr.format.bitsPerSample != 16 || hdr.format.chanNo != 1)
    throw std::runtime_error("Only 16 bit mono PCMs are supported");
  // Seek data chunk at the end of previous chunk header + chunkSize
  auto* dataHeader = reinterpret_cast<wavChunkHeader*>(reinterpret_cast<char*>(&hdr.format.format) + hdr.format.chunkHeader.chunkSize);
  // Validate data chunk id
  if (!isChunkIDEqual(dataHeader->chunkID, "data"))
    throw std::runtime_error("Corrupted data header");

  Waveform rc(dataHeader->chunkSize / sizeof(short), hdr.format.samplesPerSec);
  auto* shorts = rc.shortData();
  ifs.seekg(std::distance(reinterpret_cast<char*>(&hdr), reinterpret_cast<char*>(dataHeader + 1)), ifs.beg);
  if (!ifs.read(reinterpret_cast<char*>(shorts), dataHeader->chunkSize))
    throw std::runtime_error("Premature end of file");
  return rc;
}
