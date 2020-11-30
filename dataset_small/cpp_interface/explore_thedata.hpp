#include<iostream>
#include<string>
#include<numeric>
#include<vector>
#include<fstream>
#include "Waveform.h"
#include<dirent.h>
#include<sys/types.h>
#include<algorithm>
#include<fstream>
#include<cstring>
using namespace std;

class PixelShiftSound

{
private:
    const char* wav_path;  // wav文件的路径
    const char* label_path; // 标签文件的路径
    int datatype;   // 使用的数据库类型
    double frame_length; // 每一帧的长度
    double frame_shift; // 每一帧的偏移
    vector<vector<double>> wav_data;  // wav音频数据
    vector<vector<double>> label_data; // label数据

public:
    PixelShiftSound (int datatype);  // 构造函数
    ~PixelShiftSound();   // 析构函数

    void get_dir_files(const char* path,vector<string>&file_list); // Get files containing in certain dir
    vector<vector<double>> spilt_strings(string path); // Spilt Strings
    vector<vector<double>> get_wav_data();   // 得到所有的wav数据
    vector<vector<double>> get_label_data(); // 得到所有的 label的data

};

bool comp1(const string &str1,const string &str2){

   return atof(str2.substr(str2.length()-8,4).c_str())>atof(str1.substr(str1.length()-8,4).c_str());
}

PixelShiftSound
::PixelShiftSound(int dp){
   this->datatype = dp;
   switch (dp)
  {
  case 0:
      this->wav_path = "../10only/S0002/train/S0002";
      this->label_path = "../10only/labels";
      this->frame_length=1.0e-2;
      this->frame_shift =0;
      break;
  case 1:
      this->wav_path = "../16only/S0002/train/S0002";
      this->label_path = "../16only/labels";
      this->frame_length=1.6e-2;
      this->frame_shift =0;
      break;
  case 2:
      this->wav_path = "../16_8/S0002/train/S0002";
      this->label_path = "../16_8/labels";
      this->frame_length=1.6e-2;
      this->frame_shift =0.8e-2;
      break;
  default:
      break;
  }


}

PixelShiftSound
::~PixelShiftSound(){
    cout<<"PixelShift Object destoryed"<<endl;
}


void PixelShiftSound::get_dir_files(const char* path,vector<string>&file_list){
    struct dirent *entry1;
    DIR *dir1 = opendir(path);
          if (dir1==NULL){
        return;
    }
    while ((entry1=readdir(dir1)) != NULL)
    {
       string a;
       a= entry1->d_name;
       if (a.length()>5){
           file_list.push_back(a);
       }   
    }
    sort(file_list.begin(),file_list.end(),comp1);
    closedir(dir1);
}

vector<vector<double>> PixelShiftSound::spilt_strings(string path){
    ifstream infile;
    infile.open(path,ios::in);
    vector<vector<double>>myvec;

    while(! infile.eof()){
        string data;
        vector<double> double_vec;
        infile>>data;
        const int length =data.length();
        char * c;
        c = new char[length+1];
        strcpy(c,data.c_str());
        char *p = strtok(c,",");
    while (p!=NULL){   
        string a =p;
        double_vec.push_back(atof(a.c_str()));
        p = strtok(NULL,",");
    }
    if (double_vec.size()==15){

        myvec.push_back(double_vec);
    }
    }
     
      return myvec;
}

vector<vector<double>> PixelShiftSound::get_wav_data(){
   vector<string> wav_vector; // wav_file_vector;
   get_dir_files(this->wav_path,wav_vector); //  Get all Wav_File_NAME
   
   
   for (int i =0;i<wav_vector.size();i++){
       
       Waveform wav;
       std::stringstream filedetails;
       filedetails<<wav_path<<"/"<<wav_vector[i];
       string wavfile_path= filedetails.str(); // 得到了wav文件的完整路径
       
       wav = loadWAV(wavfile_path);
       
       // 设置 frame length 和 frame shift length
       auto bufferSize = static_cast<unsigned int>(wav.sampleRate*this->frame_length);
       auto buffer_shift = static_cast<unsigned int>(wav.sampleRate*this->frame_shift);
       
       vector<vector<double>> frame_data_file;
         
       // Get the data
       if(this->datatype!=2)   // 16only or 10only
       {
        for(auto offs(0u);offs+bufferSize<=wav.sampleCount;offs+=bufferSize){
            float* start_ptr = wav.floatData()+offs;
            vector<double> frame_data;
        for(int i=0;i<bufferSize;i++){
            frame_data.push_back(*(start_ptr+i));
                }
          frame_data_file.push_back(frame_data);
        }
       }
       else                          // 16-8 
       {   //首先加入第一帧
           auto offs(0u);
           if((offs+bufferSize)<wav.sampleCount){
               float *start_ptr = wav.floatData();
               vector<double> frame_data;
                for(int i =0;i<bufferSize;i++){
                    frame_data.push_back(*(start_ptr+i));
                }
            frame_data_file.push_back(frame_data);
           }
           //然后加上其他的帧
           for(auto offs =buffer_shift;offs+bufferSize<=wav.sampleCount;offs+=buffer_shift){
                float* start_ptr = wav.floatData()+offs;
                vector<double> frame_data;
           for(int i=0;i<bufferSize;i++){
            frame_data.push_back(*(start_ptr+i));
                }
             frame_data_file.push_back(frame_data);
           }
       }
       
      // Push All the data into a BIG vector

         for(int i=0;i<frame_data_file.size();i++){
         this->wav_data.push_back(frame_data_file[i]);
         }
   }
   return this->wav_data;

}

vector<vector<double>> PixelShiftSound::get_label_data(){
   
   vector<string>label_vector;
   get_dir_files(label_path,label_vector);

   // Then get the wav label, Save it into a big vector<vector>
   for(int i=0;i<label_vector.size();i++){
    std::stringstream stream;
    stream<<label_path<<"/"<<label_vector[i];
    string label_file_name = stream.str();
    vector<vector<double>> label_data_e = spilt_strings(label_file_name);
    for(int j =0;j<label_data_e.size();j++){
            this->label_data.push_back(label_data_e[j]);
        }
   }

   return this->label_data;
}
