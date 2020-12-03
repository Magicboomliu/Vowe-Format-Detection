# Vowe-Format-Detection


* The Python verison requires:**numpy, sklearn ,librosa**  
* The C++ version under the warped **libSVM**.  

##### For the LibSVM C++ verison, Make Sure you have CMake Tools. First Build the target, Run it in 'build' directory name 'train'  

* **Lipsync Oculus tools are provided, you can use Native C++ version or the Unity Version. Please note that the usage of this API can only run on MacOs,Win64,Android**.  


But the Linux is not supported yet for the reason that the Oculus did not give the source code which makes it hard to compile to .so or .a library.  

### **Lastest:**  
A small dataset annoated by OVRLipSync was provided, both C++ and Python Interface were provided, You can use this dataset for vowel classification using the 
algorithm you like: Deep Learning, SVM, Decision Tree, etc.  
**NOTE**:The data Label is 15 dim vector, From 1 to 15 represents the probablity of the class belongings. They are:  
* sil  
* PP  
* FF  
* TH  
* DD  
* kk  
* CH  
* SS  
* nn  
* RR  
* aa  
* E  
* ih  
* oh  
* ou  
Each Viseme corresponses to a certain mouth motion:  
![](https://github.com/Magicboomliu/Vowe-Format-Detection/blob/main/New%20Unity%20Project/Assets/Oculus/LipSync/Models/FemaleHead_Morph/Selection_034.png)  

Once You donwload the small dataset, You can use C++ or Python Interface to via the data :  
#### C++ Example: 
```
 
#include<iostream>
#include<vector>
#include "explore_thedata.hpp"
using namespace std;


int main(int argc, char const *argv[])
{

  // O represents the datatype is 0:
  // There are three datatype: 0,1,2
  // 0 :  Frame length is 10ms, frame shift is 0 .Sample rate is 16Khz, wav is 16bit.
   // 1 :  Frame length is 16ms, frame shift is 0 .Sample rate is 16Khz, wav is 16bit.
    // 0 :  Frame length is 16ms, frame shift is 8ms .Sample rate is 16Khz, wav is 16bit.
  
    PixelShiftSound ps(0);
    vector<vector<double>> wav_data;
    wav_data = ps.get_wav_data();
    cout<<wav_data.size()<<endl;

    vector<vector<double>> label_data;
    label_data = ps.get_label_data();
    cout<<label_data.size()<<endl;

}
```  
#### Python Example  (Make sure you have install Librosa and numpy):

```
pip install librosa  pip install numpy
```  
```
import  os
import  numpy as np
import librosa
import glob

from explore_data import PixelShiftSound

if __name__ == "__main__":
    '''
    datatype = 0 : 10 only :frame length is 10ms， No frame shift, sample rate is 16K
    datatype = 1 : 16 only :frame length is 16ms,  No frame shift, sample rate is 16K 
    datatype = 2 : 16-8:  frame length is 16ms, frame shift is 8ms. sample rate is 16K 
    
    '''
    ps = PixelShiftSound(sample_rate=16000,frame_duration=0.010,frame_shift_duration=0,datatype=0)
    wav_data,wav_label = ps.get_all_wav_data()
    print("Wav Frame data：",wav_data.shape)
    print("Wav Frame label：",wav_label.shape)
```

Then try to use the data do something different.


