#ifndef BLOBGETTER_H
#define BLOBGETTER_H
#include "opencv2/opencv.hpp"
#include "timedispersion.h"
#include "vibe.h"
enum BackgroudSubstractionTechique {
    TIMEDISPERSION, VIBE
};

class BlobGetter
{
public:
    BlobGetter(BackgroudSubstractionTechique backgroundSubtractor = TIMEDISPERSION);
    void Process(cv::Mat input, cv::Mat& skinMap, cv::Mat &foregroundMap);

private:
    void InitializeBackgroundSubtractor(cv::Mat firstFrame);
    void GetForegroundMap(cv::Mat input, cv::Mat *foregroundMap);
    void GetSkinRegionMap(cv::Mat input, cv::Mat *skinMap);
    BackgroudSubstractionTechique backgroundSubtractor;
    TimeDispersion timeDispersion;
    ViBe vibe;
    bool isFirstFrame;
};

#endif // BLOBGETTER_H
