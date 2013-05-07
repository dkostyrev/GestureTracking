#ifndef BLOBINTEGRALHISTOGRAM_H
#define BLOBINTEGRALHISTOGRAM_H
#include "opencv2/opencv.hpp"
#include "math.h"
#include "IntegralOrientationHistogram.h"

class BlobIntegralHistogram
{
public:
    BlobIntegralHistogram(size_t sectors, cv::Mat blobMask, cv::Point histCenter);
    BlobIntegralHistogram(std::vector<float> histogram);
    void Calculate();
    void Plot();
    cv::Mat circularHistogram;
    std::vector<sector> histogram;
private:
    void InitializeHistogram();
    void NormalizeHistogram();
    void AddToHistogram(double angle);
    cv::Mat blobMask;
    cv::Point histCenter;
    int sectors;
    size_t area;
};

#endif // BLOBINTEGRALHISTOGRAM_H
