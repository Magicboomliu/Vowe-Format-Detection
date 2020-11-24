cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)

project(ProcessWAV CXX)

add_executable(ProcessWAV ProcessWAV.cpp Waveform.h)
target_include_directories(ProcessWAV PUBLIC ../../Include)
if (${CMAKE_HOST_APPLE})
  find_library(OVRLipSyncShim OVRLipSyncShim HINT ../../Lib/MacOS)
  target_link_libraries(ProcessWAV PUBLIC ${OVRLipSyncShim})
endif ()

if (${CMAKE_HOST_WIN32})
  find_library(OVRLipSyncShimRel OVRLipSyncShim HINT ../../Lib/Win64)
  find_library(OVRLipSyncShimDbg OVRLipSyncShimd HINT ../../Lib/Win64)
  target_link_libraries(ProcessWAV optimized ${OVRLipSyncShimRel} debug ${OVRLipSyncShimDbg})
endif ()
