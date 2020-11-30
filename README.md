# Vowe-Format-Detection
A easy task trying to use InputAduio's MFCC， LPC, PLP, RPLP features together with a SVM method to identify the vowel in Chinsese **a.i.u.e.o**  

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



