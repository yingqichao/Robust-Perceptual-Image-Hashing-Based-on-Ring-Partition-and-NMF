#include "NMF.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

//clock_t start, endq;

int main()
{
	char* path1 = "E:/HashTest/HashTest/pic/330.jpg";
	char* path2 = "E:/HashTest/HashTest/pic/331.jpg";
	
	char* out1 = SingleHashCode(path1);
	cout << "out1 = " << out1 << endl;

	char* out2 = SingleHashCode(path2);
	cout << "out2 = " << out2 << endl;

	float sim = HashCodeCompare_string(out1, out2); //第一次比对，比较两串哈希码字符串的SSIM
	cout << "sim = " << sim << endl;

	//start = clock();
	
	int a = HashCodeCompare_second(path1, path2);  //第二次比对，仅对第一次结果大于0.9的每对图片进行SURF特征点匹配
	cout << a << endl;
	
	//endq = clock();
	//double endtime = (double)(endq - start) / CLOCKS_PER_SEC;
	//cout << "Total time:" << endtime * 1000 << "ms" << endl;	//计时
	system("pause");
	return 0;
}

int HashCodeCompare_second(char* in1, char* in2)  //第二次比对，仅对第一次结果大于0.9的每对图片进行SURF特征点匹配
{
	Mat img_1 = imread(in1);
	Mat img_2 = imread(in2);
	if (img_1.empty() || img_2.empty())      //保证路径错误时不报错
	{
		int p = 0;
		return p;
	}

	Ptr<Feature2D> Detector = xfeatures2d::SURF::create();

	vector<KeyPoint> keypoints1, keypoints2;   //得到特征点
	Detector->detect(img_1, keypoints1);
	Detector->detect(img_2, keypoints2);

	Mat descriptors1, descriptors2;         //特征点描述子
	Detector->compute(img_1, keypoints1, descriptors1);
	Detector->compute(img_2, keypoints2, descriptors2);

	FlannBasedMatcher matcher;               //特征点匹配
	vector<vector<DMatch>> matchpoints;
	vector<DMatch> Usefulmatchpoints;
	vector<Mat> traindescriptors(1, descriptors1);
	matcher.add(traindescriptors);
	matcher.train();
	matcher.knnMatch(descriptors2, matchpoints, 2);
	//matcher.match(descriptors1, descriptors2, matchpoints);

	for (int i = 0; i < matchpoints.size(); i++)   //选其中较好的特征点
	{
		if (matchpoints[i][0].distance < 0.5 * matchpoints[i][1].distance)
		{
			Usefulmatchpoints.push_back(matchpoints[i][0]);
		}
	}

	int size1 = min(img_1.rows, img_1.cols);
	int size2 = min(img_2.rows, img_2.cols);
	int i = 0;
	
	if (size1 >= 300 && size2 >= 300 )   //以下均为根据不同图片尺寸选取阈值，阈值用于判别匹配点数量是否足够
	{
		if (Usefulmatchpoints.size() > 100)
		{
			i = 1;
		}
		else 
		{
			i = 0;
		}
	}

	else if (size1 < 150 || size2 < 150) 
	{
		if (Usefulmatchpoints.size() > 10)
		{
			i = 1;
		}
		else
		{
			i = 0;
		}
	}

	else
	{
		if (Usefulmatchpoints.size() > 30)
		{
			i = 1;
		}
		else
		{
			i = 0;
		}
	}
	cout << Usefulmatchpoints.size() << endl;
	return i;
}

float HashCodeCompare_string(char* hash1, char* hash2)  //比较两个字符串输入的相关性
{
	string  line = hash1;
	float* tmp = new float[64];
	string key = line;
	int p = 0;
	for (int i = 0; i < 64; i++) {
		int t = 0;
		for (int j = 0; j < 3; j++) {
			t += line[3 * i + j] - '!';
			p++;
		}
		tmp[i] = (float)t;
	}

	string  line2 = hash2;
	float* tmp2 = new float[64];
	string key2 = line2;
	int p2 = 0;
	for (int i = 0; i < 64; i++) {
		int t = 0;
		for (int j = 0; j < 3; j++) {
			t += line2[3 * i + j] - '!';
			p++;
		}

		tmp2[i] = (float)t;
	}

	
	for (int i = 0; i < 64; i++)
	{
		cout << tmp[i] ;
		cout << ";";
	}
	cout << endl;

	for (int i = 0; i < 64; i++)
	{
		cout << tmp2[i];
		cout << ";";
	}
	cout << endl;
		
	//float sim = corr(tmp, tmp2, 64);
	float sim = ssim(tmp, tmp2, 64);
	delete []tmp;
	delete []tmp2;
	return sim;
}

char* SingleHashCode(char* in)    //单张图像生成Hash码返回string
{
	
	Mat oriImg = imread(in);
	char* p;
	if (oriImg.empty())
	{
		p = "\0";
		return p;
	}
	int imgsize = 512;
	int ring = 32;
	int rank = 2;

	Mat H1 = RingNMF(oriImg, imgsize, ring, rank, in);
	int len = H1.rows*H1.cols; //len1==len2
	uchar* f1 = new uchar[len];
	int k = 0;
	for (int j = 0; j < H1.cols; j++)
	{
		for (int i = 0; i < H1.rows; i++)
		{
			f1[k] = round(H1.at<float>(i, j));
			k++;
		}
	}
	map<string, uchar*> m1;
	m1.insert(make_pair(in, f1));
	delete []f1;

	map<string, uchar*>::iterator iter;
	iter = m1.begin();
	string out;
	while (iter != m1.end()) {
		string name = iter->first;
		uchar* h = iter->second;
		for (int i = 0; i < len; i++) {
			int t = (int)h[i];
			for (int j = 0; j < 3; j++) {
				int cur = min(93, t);
				char tmp = '!' + cur;
				t -= cur;
				out.push_back(tmp);
			}
		}
		iter++;
	}

	int hashlen = out.length();
	p = (char *)malloc((hashlen + 1) * sizeof(char));
	//char* p = nullptr;
	out.copy(p, hashlen, 0);

	//static char q[193];
	//int i;
	//for (i = 0; i < out.length(); i++)
	//	q[i] = out[i];
	//q[i] = '\0';

	//char* p = q;
	//memcpy(p, (char*)out.data(), (out.length() + 1) * sizeof(char));
	//p = (char*)out.data();
	//p = (char*)out.data();
	//cout << p << endl;
	return p;
}


void GenerateHashCode(char* in, int mode) {
	//mode:1-hashcode.txt(源文件),2-hashtest.txt(待检测文件)
	int len = 64;
	map<string, uchar*> Map = calculation(in);
	string pname = (string)in + "\\" + ((mode == 0) ? "hashcode.txt" : "hashtest.txt");
	saveFile(Map, pname, len);
}


void observeHashCode(char* in) {
	try {
		map<string, uchar*> m1;
		int len0 = 64; int imgsize = 512; // Image size
		int ring = 32; // Ring Number
		int rank = 2; // Rank
		Mat oriIm = imread(in);
		Mat H1 = RingNMF(oriIm, imgsize, ring, rank, in);
		int len = H1.rows*H1.cols; //len1==len2
		uchar* f1 = new uchar[len];
		int k = 0;
		for (int j = 0; j < H1.cols; j++)
		{
			for (int i = 0; i < H1.rows; i++)
			{
				//float转uchar
				f1[k] = round(H1.at<float>(i, j));
				k++;
			}
		}
		m1.insert(make_pair(in, f1));
		string pname = "observe.txt";
		saveFile(m1, pname, len0);
	}
	catch (cv::Exception& e) {
		cout << "Error loading Image: " << in << endl;
	}

}

void HashTest(char* in, char* fold2, int tolerance) {
	int len0 = 64;
	ofstream   ofresult("result.txt");

	if (!ofresult.is_open()) {
		cout << "File is open fail!" << endl;
		return;
	}

	map<string, float*> Map1;
	int imgsize = 512; // Image size
	int ring = 32; // Ring Number
	int rank = 2; // Rank
	Mat oriIm = imread(in);
	Mat H1 = RingNMF(oriIm, imgsize, ring, rank, in);
	int len = H1.rows*H1.cols;     //len1==len2
	float* f1 = new float[len];
	int k = 0;
	for (int j = 0; j < H1.cols; j++)
	{
		for (int i = 0; i < H1.rows; i++)
		{
			//float转uchar
			f1[k] = round(H1.at<float>(i, j));
			k++;
		}
	}
	Map1.insert(make_pair(in, f1));


	map<string, float*> Map2 = read("hashtest.txt", len0);
	map<string, float*>::iterator iter1;
	iter1 = Map1.begin();
	while (iter1 != Map1.end()) {
		string targetname = iter1->first;
		cout << "Comparing Result For :" << targetname << endl;
		ofresult << "Compare Result For :" << targetname << endl;
		float* F1 = iter1->second;
		map<string, float*>::iterator iter2;
		iter2 = Map2.begin();
		while (iter2 != Map2.end()) {
			string comparename = iter2->first;
			float* F2 = iter2->second;

			float sim = corr(F1, F2, len);
			if (sim >= tolerance / 10.0) {
				ofresult << sim << " " << targetname << endl;
				cout << sim << " " << targetname << endl;
			}
			iter2++;
		}
		iter1++;
	}

	ofresult.close();
}

void HashTest_folder(char* fold1, char* fold2, int tolerance) {
	int len = 64;
	ofstream   ofresult("result.txt");

	if (!ofresult.is_open()) {
		cout << "File is open fail!" << endl;
		return;
	}
	map<string, float*> Map1 = read("hashcode.txt", len);
	map<string, float*> Map2 = read("hashtest.txt", len);
	map<string, float*>::iterator iter1;
	iter1 = Map1.begin();
	while (iter1 != Map1.end()) {
		string targetname = iter1->first;
		cout << "Comparing Result For :" << targetname << endl;
		ofresult << "Compare Result For :" << targetname << endl;
		float* F1 = iter1->second;
		map<string, float*>::iterator iter2;
		iter2 = Map2.begin();
		while (iter2 != Map2.end()) {
			string comparename = iter2->first;
			float* F2 = iter2->second;

			float sim = corr(F1, F2, len);
			if (sim >= tolerance / 10.0) {
				ofresult << sim << " " << targetname << endl;
				cout << sim << " " << targetname << endl;
			}
			iter2++;
		}
		iter1++;
	}

	ofresult.close();
}


/**
vector<string> findTxt(char* in_path) {
	string str = in_path;
	struct _finddata_t fileinfo;
	vector<string> res;
	string in_name;

	string curr = str + "\\*.txt";
	long handle;
	if ((handle = _findfirst(curr.c_str(), &fileinfo)) == -1L)
	{
		cout << "没有找到匹配文件!" << endl;
	}
	else
	{
		in_name = str + "\\" + fileinfo.name;
		res.push_back(in_name);
		while (!(_findnext(handle, &fileinfo)))
		{
			in_name = str + "\\" + fileinfo.name;
			res.push_back(in_name);
		}
		_findclose(handle);
	}
	return res;
}
**/

map<string, uchar*> calculation(char* fold) {
	//Settings
	int imgsize = 512; // Image size
	int ring = 32; // Ring Number
	int rank = 2; // Rank

	map<string, uchar*> m1;
	string format[6] = { "/*.jpg","/*.png","/*.tif","/*.bmp","/*.jpeg","/*.tiff" };
	for (int ind = 0; ind < 6; ind++) {
		cv::String pattern1 = fold + format[ind];

		vector<cv::String> fn1;
		glob(pattern1, fn1, false);

		for (int im = 0; im < fn1.size(); im++) {
			try {
				Mat oriIm = imread(fn1[im]);
				Mat H1 = RingNMF(oriIm, imgsize, ring, rank, fn1[im]);
				int len = H1.rows*H1.cols; //len1==len2
				uchar* f1 = new uchar[len];
				int k = 0;
				for (int j = 0; j < H1.cols; j++)
				{
					for (int i = 0; i < H1.rows; i++)
					{
						//float转uchar
						f1[k] = round(H1.at<float>(i, j));
						k++;
					}
				}
				m1.insert(make_pair(fn1[im], f1));
			}
			catch (cv::Exception& e) {
				cout << "Error loading Image: " << fn1[im] << endl;
			}
		}
	}

	return m1;
}



Mat RingNMF(Mat Im, int imgsize, int ring, int rank, string name) {
	//对图像I进行环形分割，
	//规格化图像为imgsize*imgsize, n个环形等面积,
	//图像内切圆的半径为R，如果按半径等分，则第i个圆环的半径 ri ^ 2 = i * R ^ 2 / n;
	resize(Im, Im, Size(imgsize, imgsize));
	cout << "Generating hashcode for: " << name << endl;
	GaussianBlur(Im, Im, Size(3, 3), 1); //低通滤波
	cvtColor(Im, Im, 37);//转YCbCr
	Mat ImChannel[3];
	//src为要分离的Mat对象  
	split(Im, ImChannel);              //利用数组分离  
	Mat I1 = ImChannel[0];
	//imshow("portion",I1);
	//waitKey();
	double cc = 0;
	if (imgsize % 2 == 0)
		cc = imgsize / 2.0 + 0.5;   //圆心坐标（cc, cc)
	else
		cc = (imgsize + 1) / 2.0;

	Mat H = cv::Mat::zeros(ring, 1, CV_32FC1);
	Mat rpt = cv::Mat::zeros(1, imgsize, CV_32FC1);

	for (int i = 0; i < imgsize; i++)
		rpt.at<float>(0, i) = i + 1;



	Mat XA = repeat(rpt, imgsize, 1);
	Mat YA; transpose(XA, YA);

	for (int j = 0; j < imgsize; j++)
	{
		for (int i = 0; i < imgsize; i++)
		{
			XA.at<float>(j, i) = pow((XA.at<float>(j, i) - cc), 2) + pow((YA.at<float>(j, i) - cc), 2);
		}
	}

	Mat RN = cv::Mat::zeros(1, ring, CV_32FC1);

	for (int i = 0; i < ring; i++)
		RN.at<float>(0, i) = i + 1;

	RN.at<float>(0, ring - 1) = imgsize - cc;

	float s = floor(CV_PI*RN.at<float>(0, ring - 1) * RN.at<float>(0, ring - 1) / ring);

	RN.at<float>(0, 0) = sqrt(s / CV_PI);

	for (int i = 1; i < ring - 1; i++)
		RN.at<float>(0, i) = sqrt((s + CV_PI * RN.at<float>(0, i - 1) * RN.at<float>(0, i - 1)) / CV_PI); //radius of each circle

	for (int i = 0; i < ring; i++)
		RN.at<float>(0, i) = pow(RN.at<float>(0, i), 2);

	Mat V = cv::Mat::zeros(s, ring, CV_32FC1);
	vector<vector<float>> vectorHolder;
	vector<float> v;
	for (int j = 0; j < imgsize; j++)
	{
		for (int i = 0; i < imgsize; i++)
		{
			if (XA.at<float>(j, i) <= RN.at<float>(0, 0)) {
				v.push_back((float)I1.at<uchar>(j, i));
			}
		}
	}
	vectorHolder.push_back(v);
	int len = v.size(); int row = 0;

	/*float* mapped = new float[s];
	funcResampling(mapped, s, &v[0], v.size());

	sort(mapped, mapped+(int)s);

	for (int i = 0; i < (int)s;i++) {
	V.at<float>(row,0) = mapped[i];
	row++;
	}*/


	//Starting from 2nd ring
	for (int r = 0; r < ring - 1; r++) {
		//cout << "Ring" << endl;
		vector<float> v1;
		for (int j = 0; j < imgsize; j++)
		{
			for (int i = 0; i < imgsize; i++)
			{
				if (XA.at<float>(j, i) <= RN.at<float>(0, r + 1) && XA.at<float>(j, i) > RN.at<float>(0, r))
					v1.push_back((float)I1.at<uchar>(j, i));
			}
		}
		len = min(len, (int)v1.size());
		vectorHolder.push_back(v1);

	}

	for (int r = 0; r < ring; r++) {
		/*funcResampling(mapped, s, &v[0], v.size());*/
		//cout << "Sort" << endl;
		vector<float> tmp = vectorHolder[r];
		sort(tmp.begin(), tmp.begin() + len);

		row = 0;

		for (int i = 0; i < len; i++) {
			V.at<float>(row, r + 1) = vectorHolder[r][i];
			row++;
		}
	}
	//for (int r = 0; r < ring - 1; r++)
	//	delete vectorHolder[r];
	//delete vectorHolder;

	int maxiter = 60;

	H = NMFLee2000(V, rank, maxiter);

	return H;
}
//
Mat NMFLee2000(Mat V, int rank, int maxiter) {
	int n = V.rows; int m = V.cols;
	cv::RNG rnger;
	cv::Mat W;
	// CV_32FC1 uniform distribution
	W.create(n, rank, CV_32FC1);
	rnger.fill(W, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(1.));
	////Reset
	//rnger(cv::getTickCount());
	cv::Mat H;
	// CV_32FC1 uniform distribution
	H.create(rank, m, CV_32FC1);
	rnger.fill(H, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(1.));

	for (int iter = 0; iter < maxiter; iter++) {
		//Stage 1

		//  %H = (H.*(W'*(V./(W*H))))./((sum(W)')*ones(1, m))
		//	%1.e1 = (W*H) s1 = (sum(W)')
		Mat e1 = W * H;
		Mat s1 = cv::Mat::zeros(rank, 1, CV_32FC1);
		for (int j = 0; j < rank; j++)
		{
			for (int i = 0; i < n; i++)
			{
				s1.at<float>(j, 0) = s1.at<float>(j, 0) + W.at<float>(i, j);
			}
		}
		//	%2.e2 = V. / e1 s2 = s1 * ones(1,m)
		Mat Ones = Mat::ones(1, m, CV_32FC1);
		Mat s2 = s1 * Ones;
		Mat e2 = Mat::zeros(V.rows, V.cols, CV_32FC1);
		for (int j = 0; j < V.rows; j++)
		{
			for (int i = 0; i < V.cols; i++)
			{
				e2.at<float>(j, i) = V.at<float>(j, i) / (e1.at<float>(j, i) + 0.0001);
			}
		}
		//	%3.e3 = W'*e2 e4 = H.*e3
		Mat W1; transpose(W, W1);
		Mat e3 = W1 * e2;
		Mat e4 = Mat::zeros(H.rows, H.cols, CV_32FC1);
		for (int j = 0; j < H.rows; j++)
		{
			for (int i = 0; i < H.cols; i++)
			{
				e4.at<float>(j, i) = H.at<float>(j, i) * e3.at<float>(j, i);
			}
		}
		//	%4.H = e4. / s2
		for (int j = 0; j < H.rows; j++)
		{
			for (int i = 0; i < H.cols; i++)
			{
				H.at<float>(j, i) = e4.at<float>(j, i) / (s2.at<float>(j, i) + 0.0001);
			}
		}

		//Stage 2

		//%1.e1 = W * H s1 = (sum(H, 2)')
		e1 = W * H;
		s1 = cv::Mat::zeros(1, rank, CV_32FC1);
		for (int j = 0; j < rank; j++)
		{
			for (int i = 0; i < m; i++)
			{
				s1.at<float>(0, j) += H.at<float>(j, i);
			}
		}
		//%2.e2 = V. / e1 s2 = ones(n, 1)*s1
		for (int j = 0; j < V.rows; j++)
		{
			for (int i = 0; i < V.cols; i++)
			{
				e2.at<float>(j, i) = V.at<float>(j, i) / (e1.at<float>(j, i) + 0.0001);
			}
		}
		Ones = Mat::ones(n, 1, CV_32FC1);
		s2 = Ones * s1;
		//%3.e3 = e2 * H' e4 = W.*e3
		Mat H1; transpose(H, H1);
		e3 = e2 * H1;
		e4 = Mat::zeros(W.rows, W.cols, CV_32FC1);
		for (int j = 0; j < W.rows; j++)
		{
			for (int i = 0; i < W.cols; i++)
			{
				e4.at<float>(j, i) = W.at<float>(j, i) * e3.at<float>(j, i);
			}
		}
		//%4.W = e4. / s2
		for (int j = 0; j < W.rows; j++)
		{
			for (int i = 0; i < W.cols; i++)
			{
				W.at<float>(j, i) = e4.at<float>(j, i) / (s2.at<float>(j, i) + 0.0001);
			}
		}

	}

	return H;
}

void funcResampling(float *pDst, unsigned destLen, float *pSrc, unsigned srcLen) {
	for (unsigned indexD = 0; indexD < destLen; indexD++)
	{
		unsigned nCount = 0;
		for (unsigned j = 0; j < srcLen; j++)
		{
			unsigned indexM = indexD * srcLen + j;
			unsigned indexS = indexM / destLen;
			nCount += pSrc[indexS];
		}
		pDst[indexD] = nCount / (float)srcLen;
	}
}

float corr(float* x, float* y, int N) {
	float sum_sq_x = 0;
	float sum_sq_y = 0;
	float sum_coproduct = 0;
	float mean_x = x[0];
	float mean_y = y[0];
	float sweep;
	float delta_x;
	float delta_y;
	for (int i = 1; i < N; i++) {
		sweep = (i - 1.0) / i;
		delta_x = x[i] - mean_x;
		delta_y = y[i] - mean_y;
		sum_sq_x += delta_x * delta_x * sweep;
		sum_sq_y += delta_y * delta_y * sweep;
		sum_coproduct += delta_x * delta_y * sweep;
		mean_x += delta_x / i;
		mean_y += delta_y / i;
	}
	float pop_sd_x = sqrt(sum_sq_x / N);
	float pop_sd_y = sqrt(sum_sq_y / N);
	float cov_x_y = sum_coproduct / N;
	float correlation = cov_x_y / (pop_sd_x * pop_sd_y);

	return correlation;
}

float ssim(float* x, float* y, int N) {
	float mean_x = 0;
	float mean_y = 0;
	float sigma_x = 0;
	float sigma_y = 0;
	float sigma_xy = 0;
	float C1 = 6.5025, C2 = 58.5225;

	for (int i = 0; i < N; i++) {
		mean_x += x[i];
		mean_y += y[i];
	}
	mean_x = mean_x / N;
	mean_y = mean_y / N;

	for (int i = 0; i < N; i++) {
		sigma_x += (x[i] - mean_x)*(x[i] - mean_x);
		sigma_y += (y[i] - mean_y)*(y[i] - mean_y);
		sigma_xy += abs((x[i] - mean_x)*(y[i] - mean_y));
	}
	sigma_x = sigma_x / N;
	sigma_y = sigma_y / N;
	sigma_xy = sigma_xy / N;
	
	float x1 = (2 * mean_x*mean_y + C1) * (2 * sigma_xy + C2);
	float x2 = (mean_x*mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
	float ssim = x1 / x2;
	return ssim;
}

void saveFile(map<string, uchar*> hashcode, string foldpath, int len) {

	ofstream   ofresult(foldpath);

	if (!ofresult.is_open()) {
		cout << "File is open fail!" << endl;
		return;
	}

	map<string, uchar*>::iterator iter;
	iter = hashcode.begin();
	while (iter != hashcode.end()) {
		string name = iter->first;
		uchar* h = iter->second;
		ofresult << name << endl;
		for (int i = 0; i < len; i++) {
			int t = (int)h[i];
			for (int j = 0; j < 3; j++) {
				int cur = min(93, t);
				char tmp = '!' + cur;
				t -= cur;
				ofresult << tmp;
			}
		}
		ofresult << endl;
		iter++;
	}

	/*while (iter != obm.end()) {
	ofresult << iter->first << endl;
	ofresult << iter->second << endl;
	iter++;
	}*/
	ofresult.close();
}

map<string, float*> read(string name, int len) {
	//说明：hashcode用来读取原图，hashtest用来读取测试图
	map<string, float*> dict;
	fstream  f(name);

	f.open(name);
	if (f.is_open())
	{

		string  line;
		while (getline(f, line))
		{
			float* tmp = new float[len];
			string key = line;
			getline(f, line);
			int p = 0;
			for (int i = 0; i < len; i++) {
				int t = 0;
				for (int j = 0; j < 3; j++) {
					t += line[3 * i + j] - '!';
					p++;
				}

				tmp[i] = (float)t;
			}
			dict.insert(make_pair(key, tmp));
		}
		f.close();
	}
	//char* ShowArr = new char[len];
	//fstream  f(name);
	//string  line;
	//getline(f, line);
	//int i = 0;

	//while (!f.eof())

	//{
	//	f >> ShowArr[i];
	//	i++;
	//}
	//f.close();
	return dict;
}