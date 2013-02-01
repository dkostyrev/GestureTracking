#include "timedispersion.h"

TimeDispersion::TimeDispersion()
{
}

TimeDispersion::TimeDispersion(int width, int height, size_t historyDepth, int threshold)
{
    history = std::vector<cv::Mat>();
    this->threshold = threshold;
    this->historyDepth = historyDepth;
    mdTime = cv::Mat(height, width, CV_32FC1);
    sqmdTime = cv::Mat(height, width, CV_32FC1);
}

void TimeDispersion::Process(cv::Mat &segMap)
{
    mdTime.setTo(0.0);
    sqmdTime.setTo(0.0);
    cv::Mat sq;
    for (size_t i = 0; i < history.size(); ++i){
        mdTime += history.at(i);
        cv::pow(history.at(i), 2, sq);
        sqmdTime += sq;
    }

    mdTime /= (float)(this->historyDepth);
    sqmdTime /= (float) (this->historyDepth);
    cv::pow(mdTime, 2, mdTime);
    segMap = sqmdTime - mdTime;
    segMap = (segMap >= this->threshold * this->threshold);
}

bool TimeDispersion::UpdateHistory(cv::Mat newFrame)
{
    newFrame.convertTo(newFrame, CV_32F);
    history.insert(history.begin(), newFrame);
    if (history.size() > this->historyDepth){
        history.erase(history.end() - 1, history.end());
        return true;
    }
    else {
        return false;
    }
}
