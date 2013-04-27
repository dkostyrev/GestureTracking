#ifndef MOTIONESTIMATOR_H
#define MOTIONESTIMATOR_H
#include "opencv2/opencv.hpp"
#include <vector>
#include "blobintegralhistogram.h"
class MotionEstimator
{
public:
    MotionEstimator();
    void AddFrame(cv::Mat frame);
    void ShowAllFrames();
    void GetMotionMat(cv::Mat& result);
    void getMotionVector(cv::Mat result, size_t framesSize);
    void calculateMotionHistograms();
private:
    std::vector<cv::Mat> frames;
};

#endif // MOTIONESTIMATOR_H
