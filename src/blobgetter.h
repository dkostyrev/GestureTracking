#ifndef BLOBGETTER_H
#define BLOBGETTER_H
#include "opencv2/opencv.hpp"
#include "timedispersion.h"
#include "vibe.h"
#include "ColorFinder.h"
enum BackgroudSubstractionTechique {
    TIMEDISPERSION, VIBE, MOG, NO
};

class BlobGetter
{
public:
    BlobGetter(BackgroudSubstractionTechique backgroundSubtractor = TIMEDISPERSION);
    void Process(cv::Mat input, cv::Mat& skinMap, cv::Mat &foregroundMap);
    void FilterBySize(cv::Mat rawSkinMap, cv::Mat &skinMap);
    void AdaptColourThresholds(cv::Mat input, cv::Rect roi);
    void ResetColourThresholds();
    void DispMap(cv::Mat input, cv::Mat &output, int threshold);
    void getForegroundMap(cv::Mat input, cv::Mat &output);
    void getMixedMap(cv::Mat input, cv::Mat &output);
private:
    cv::Scalar defaultLow, defaultHigh, adaptLow, adaptHigh;
    bool isThresholdsAdapted;
    void InitializeBackgroundSubtractor(cv::Mat firstFrame);
    void GetForegroundMap(cv::Mat input, cv::Mat &foregroundMap);
    void GetSkinRegionMap(cv::Mat input, cv::Mat &skinMap);
    void MedianFilter(cv::Mat input, cv::Mat &output, size_t times);
    BackgroudSubstractionTechique backgroundSubtractor;
    TimeDispersion timeDispersion;
    ViBe vibe;
    cv::BackgroundSubtractorMOG2 mog2;
    cv::Mat medianKernel;
    bool isFirstFrame;
};

#endif // BLOBGETTER_H
