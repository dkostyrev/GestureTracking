#pragma once
#include "opencv2/opencv.hpp"
#include "IntegralOrientationHistogram.h"
#include "classifier.h"
#include <iostream>
#include <fstream>
#include <math.h>

class BlobProcessor
{
public:
    BlobProcessor();
    cv::Point GetCenterOfMasses(std::vector<cv::Point> contour);
    cv::Point GetCenterOfMasses(cv::Mat& blobMask);
    double GetEccentricity(std::vector<cv::Point> contour);
};
