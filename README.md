# Vowe-Format-Detection
A easy task trying to use InputAduio's MFCC， LPC, PLP, RPLP features together with a SVM method to identify the vowel in Chinsese **a.i.u.e.o**  

The Python verison requires:**numpy, sklearn ,librosa**  
The C++ version under the warped **libSVM**.  

##### For the LibSVM C++ verison, Make Sure you have CMake Tools. First Build the target, Run it in 'build' directory name 'train'  

**Lipsync Oculus tools are provided, you can use Native C++ version or the Unity Version. Please note that the usage of this API can only run on MacOs,Win64,Android**.  


But the Linux is not supported yet for the reason that the Oculus did not give the source code which makes it hard to compile to .so or .a library.
