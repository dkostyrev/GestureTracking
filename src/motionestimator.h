#ifndef MOTIONESTIMATOR_H
#define MOTIONESTIMATOR_H
#include "opencv2/opencv.hpp"
#include <vector>
#include "blobintegralhistogram.h"
#include "direct.h"
#include "time.h"
#include "blobprocessor.h"
//#include "tbb.h"
class MotionEstimator
{
public:
    MotionEstimator();
    void AddFrame(cv::Mat frame);
    void ShowAllFrames();
    size_t GetFrameCount();
    void clear();
    void GetMotionMat(cv::Mat& result);
    void getMotionVector(cv::Mat result, size_t framesSize);
    void calculateMotionHistograms(std::vector<std::vector<double> > &histograms, bool plot, bool save);
private:
    std::vector<cv::Mat> frames;
};

#endif // MOTIONESTIMATOR_H
