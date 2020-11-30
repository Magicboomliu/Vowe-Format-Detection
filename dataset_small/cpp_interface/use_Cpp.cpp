#include<iostream>
#include<vector>
#include "explore_thedata.hpp"
using namespace std;


int main(int argc, char const *argv[])
{
    PixelShiftSound ps(2);
    vector<vector<double>> wav_data;
    wav_data = ps.get_wav_data();
    cout<<wav_data.size()<<endl;

    vector<vector<double>> label_data;
    label_data = ps.get_label_data();
    cout<<label_data.size()<<endl;

}



