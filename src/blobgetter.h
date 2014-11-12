#pragma once
#include "opencv2/opencv.hpp"
#include "timedispersion.h"
#include "vibe.h"
#include "ColorFinder.h"

class BlobGetter
{
public:

    enum BackgroudSubtractionTechique {
        TIMEDISPERSION,
        VIBE,
        MOG,
        NO
    };

    BlobGetter(BackgroudSubtractionTechique backgroundSubtractor = TIMEDISPERSION);
    void Process(cv::Mat& input, cv::Mat& skinMap, cv::Mat& foregroundMap);
    void FilterBySize(cv::Mat& rawSkinMap, cv::Mat& skinMap);
    void AdaptColourThresholds(cv::Mat& input, cv::Rect roi);
    void ResetColourThresholds();
    void DispMap(cv::Mat& input, cv::Mat& output, int threshold);
    void GetForegroundMap(cv::Mat& input, cv::Mat& output);
    void GetMixedMap(cv::Mat& input, cv::Mat& output);
private:
    void InitializeBackgroundSubtractor(cv::Mat& firstFrame);
    void MedianFilter(cv::Mat& input, cv::Mat& output, size_t times);
    void ProcessForegroundMap(cv::Mat& input, cv::Mat& output);
    cv::Scalar defaultLow, defaultHigh, adaptLow, adaptHigh;
    bool isThresholdsAdapted;
    BackgroudSubtractionTechique backgroundSubtractor;
    TimeDispersion timeDispersion;
    ViBe vibe;
    cv::BackgroundSubtractorMOG2 mog2;
    cv::Mat medianKernel;
    bool isFirstFrame;
};
