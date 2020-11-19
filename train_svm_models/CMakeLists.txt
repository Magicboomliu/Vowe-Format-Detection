#聲明cmake的使用版本
cmake_minimum_required(VERSION 2.8)
# set(CMAKE_CXX_FLAGS "-std=c++11 -fsanitize=address")
set(CMAKE_CXX_FLAGS "-std=c++11 -fsanitize=address")

#聲明一個 cmake 工程
project(Train)


#聲明一個 可執行程序
add_library(svm SHARED svm.cpp)
# 添加庫文件

add_executable(train train.cpp)
target_link_libraries(train svm)