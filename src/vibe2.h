#ifndef __bgfg_vibe_h__
#define __bgfg_vibe_h__
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"    

using namespace cv;
using namespace std;

int init_model(Mat& firstSample);
Mat* fg_vibe(Mat& frame,int idx);

#endif