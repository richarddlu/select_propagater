#include "select_propagater.h"

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool flag = false;
bool incluFlag = true;
Mat img_show;
Mat selects;
Point* pre;

void mouseCallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if(event == EVENT_LBUTTONDOWN) {
		flag = !flag;
		pre = NULL;
	}
	if(event == EVENT_RBUTTONDOWN) {
		incluFlag = !incluFlag;
	}
	if(event == EVENT_MOUSEMOVE) {
		if(flag == true) {
			if(pre != NULL) {
				Point* cur = new Point(x, y);
				if(incluFlag) {
					line(img_show, *pre, *cur, Scalar(0, 255, 0), 5);
					line(selects, *pre, *cur, Scalar(1), 5);
				} else {
					line(img_show, *pre, *cur, Scalar(0, 0, 255), 5);
					line(selects, *pre, *cur, Scalar(-1), 5);
				}
				pre = cur;
				imshow("demo", img_show);
			} else {
				pre = new Point(x, y);
			}
		}
	}
}

int main() {
	Mat img = imread("images/test.jpg");
	img_show = img.clone();

	// initialize selects matrix
	selects = Mat::zeros(img.size(), CV_8S);

	namedWindow("demo");

	setMouseCallback("demo", mouseCallBackFunc, NULL);

	imshow("demo", img);

	waitKey(0);

	// export mask
	// FileStorage fs("selects.yml", FileStorage::WRITE);
	// fs<<"selects"<<selects;

	// import mask
	// FileStorage fs("selects.yml", FileStorage::READ);
	// fs["selects"]>>selects;

	SelectPropagater sp(img, selects);
	sp.debug = true;
	sp.basisSampleMethod = Uniform;
	// sp.equSampleMethod = Uniform;
	// sp.basisSampleMethod = NoSample;
	sp.equSampleMethod = NoSample;

	Mat sMap;
	sp.apply(sMap);

	imshow("similarity map", sMap);

	cout<<"Number of Selects: "<<sp.numSelects<<endl;
	cout<<"Number of Basis: "<<sp.numBasis<<endl;
	cout<<"Number of Equations: "<<sp.numEquations<<endl;
	cout<<"MSE: "<<sp.selectsMSE<<endl;

	imshow("basis", sp.basisShow);
	imshow("equ", sp.equShow);

	waitKey(0);

	return 0;
}
