# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liuzihua/lzh_intern/index_files/train_svm_models

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuzihua/lzh_intern/index_files/train_svm_models/build

# Include any dependencies generated for this target.
include CMakeFiles/train.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/train.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/train.dir/flags.make

CMakeFiles/train.dir/train.cpp.o: CMakeFiles/train.dir/flags.make
CMakeFiles/train.dir/train.cpp.o: ../train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuzihua/lzh_intern/index_files/train_svm_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/train.dir/train.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train.dir/train.cpp.o -c /home/liuzihua/lzh_intern/index_files/train_svm_models/train.cpp

CMakeFiles/train.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train.dir/train.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuzihua/lzh_intern/index_files/train_svm_models/train.cpp > CMakeFiles/train.dir/train.cpp.i

CMakeFiles/train.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train.dir/train.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuzihua/lzh_intern/index_files/train_svm_models/train.cpp -o CMakeFiles/train.dir/train.cpp.s

CMakeFiles/train.dir/train.cpp.o.requires:

.PHONY : CMakeFiles/train.dir/train.cpp.o.requires

CMakeFiles/train.dir/train.cpp.o.provides: CMakeFiles/train.dir/train.cpp.o.requires
	$(MAKE) -f CMakeFiles/train.dir/build.make CMakeFiles/train.dir/train.cpp.o.provides.build
.PHONY : CMakeFiles/train.dir/train.cpp.o.provides

CMakeFiles/train.dir/train.cpp.o.provides.build: CMakeFiles/train.dir/train.cpp.o


# Object files for target train
train_OBJECTS = \
"CMakeFiles/train.dir/train.cpp.o"

# External object files for target train
train_EXTERNAL_OBJECTS =

train: CMakeFiles/train.dir/train.cpp.o
train: CMakeFiles/train.dir/build.make
train: libsvm.so
train: CMakeFiles/train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuzihua/lzh_intern/index_files/train_svm_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable train"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/train.dir/build: train

.PHONY : CMakeFiles/train.dir/build

CMakeFiles/train.dir/requires: CMakeFiles/train.dir/train.cpp.o.requires

.PHONY : CMakeFiles/train.dir/requires

CMakeFiles/train.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/train.dir/cmake_clean.cmake
.PHONY : CMakeFiles/train.dir/clean

CMakeFiles/train.dir/depend:
	cd /home/liuzihua/lzh_intern/index_files/train_svm_models/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuzihua/lzh_intern/index_files/train_svm_models /home/liuzihua/lzh_intern/index_files/train_svm_models /home/liuzihua/lzh_intern/index_files/train_svm_models/build /home/liuzihua/lzh_intern/index_files/train_svm_models/build /home/liuzihua/lzh_intern/index_files/train_svm_models/build/CMakeFiles/train.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/train.dir/depend

