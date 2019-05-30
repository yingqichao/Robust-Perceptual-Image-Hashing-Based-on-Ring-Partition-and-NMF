#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

//hashcodeΪ�ӣ���ʼ����λ�洢
//private:
Mat NMFLee2000(Mat V, int rank, int maxiter);
map<string, uchar*> calculation(char* fold);
map<string, float*> read(string name, int len);
Mat RingNMF(Mat Im, int imgsize, int ring, int rank, string name);
void saveFile(map<string, uchar*> hashcode, string foldpath, int len);
void funcResampling(float *pDst, unsigned destLen, float *pSrc, unsigned srcLen);
float corr(float* x, float* y, int N);
float ssim(float* x, float* y, int N);
//vector<string> findTxt(char* in_path);
//public:
void GenerateHashCode(char* in, int mode);
void HashTest_folder(char* filename, char* foldname, int tolerance);//�������ļ��м������ͼ����һһ�ȶ�
//void Test(char* A, char* B);//�ṩ�㷨������
void observeHashCode(char* in);
void HashTest(char* in, char* fold2, int tolerance);
char* SingleHashCode(char* in);
float HashCodeCompare_string(char* hash1, char* hash2);
int HashCodeCompare_second(char* in1, char* in2);
