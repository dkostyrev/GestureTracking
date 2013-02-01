#ifndef TIMEDISPERSION_H
#define TIMEDISPERSION_H
#include "opencv2/opencv.hpp"
#include <vector>
class TimeDispersion
{
public:
    TimeDispersion();
    TimeDispersion(int width, int height, size_t historyDepth, int threshold);
    void Process(cv::Mat &segMap);
    bool UpdateHistory(cv::Mat newFrame);
private:
    cv::Mat mdTime;
    cv::Mat sqmdTime;
    size_t historyDepth;
    int threshold;
    std::vector<cv::Mat> history;
};

#endif // TIMEDISPERSION_H
