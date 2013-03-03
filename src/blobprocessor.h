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

    void GetGoodClusters(std::vector<cv::Rect> clusters, std::vector<cv::Rect> &goodClusters);
    void DispMap(cv::Mat input, cv::Mat &output);
    void RecognizeClusters(cv::Mat input, std::vector<cv::Rect> clusters);
private:
    cv::Mat hsv;
};



#endif // BLOBPROCESSOR_H
