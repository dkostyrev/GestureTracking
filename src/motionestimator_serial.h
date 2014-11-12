#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include "blobintegralhistogram.h"
#include "direct.h"
#include "time.h"
#include "blobprocessor.h"
class MotionEstimator
{
public:
    void AddFrame(cv::Mat& frame);
    void ShowAllFrames();
    size_t GetFrameCount();
    void Clear();
    void GetMotionMat(cv::Mat& result);
    void GetMotionVector(cv::Mat& result, size_t framesSize);
    void CalculateMotionHistograms(std::vector<std::vector<double> > &histograms, bool plot, bool save);
private:
    std::vector<cv::Mat> frames;
};
