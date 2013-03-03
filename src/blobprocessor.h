#ifndef BLOBPROCESSOR_H
#define BLOBPROCESSOR_H
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "colorFinder.h"
class BlobProcessor
{
public:
    BlobProcessor();
    void Process(cv::Mat skinMap, cv::Mat input);

    void SuppressBadClusters(cv::Mat input, std::vector<cv::Rect> cluster);
private:
    cv::Mat hsv;
};



#endif // BLOBPROCESSOR_H
