#include "blobgetter.h"
BlobGetter::BlobGetter(BackgroudSubstractionTechique backgroundSubtractor)
{
    this->backgroundSubtractor = backgroundSubtractor;
    this->isFirstFrame = true;
    this->medianKernel = cv::Mat(9, 9, CV_32F);
    this->medianKernel.setTo(cv::Scalar(1./81));
    this->defaultLow = cv::Scalar(20, 133, 80);
    this->defaultHigh = cv::Scalar(140, 173, 120);
    ResetColourThresholds();
}

void BlobGetter::Process(cv::Mat input, cv::Mat &skinMap, cv::Mat &foregroundMap)
{
    GetSkinRegionMap(input, skinMap);
    //FilterBySize(skinMap, skinMap);
    MedianFilter(skinMap, skinMap, 1);
    cv::cvtColor(input, input, CV_BGR2GRAY);
    if (isFirstFrame){
        foregroundMap = cv::Mat(input.rows, input.cols, CV_8UC1);
        InitializeBackgroundSubtractor(input);
        isFirstFrame = false;
    }
    else {
        GetForegroundMap(input, foregroundMap);
        MedianFilter(foregroundMap, foregroundMap, 2);
    }

}

void BlobGetter::FilterBySize(cv::Mat rawSkinMap, cv::Mat& skinMap)
{
    const int THRESHOLD = 100;
    std::vector<std::vector<cv::Point> > contours, goodContours;
    cv::findContours(rawSkinMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours.at(i)) > THRESHOLD) {
            goodContours.push_back(contours.at(i));
        }
    }
    skinMap = cv::Mat(rawSkinMap.rows, rawSkinMap.cols, CV_8UC1);
    cv::drawContours(skinMap, goodContours, -1, cv::Scalar(255), CV_FILLED, CV_AA);
}


void BlobGetter::InitializeBackgroundSubtractor(cv::Mat firstFrame)
{
    switch (this->backgroundSubtractor){
        case TIMEDISPERSION: {
            timeDispersion = TimeDispersion(firstFrame.cols, firstFrame.rows, 2, 2);
            timeDispersion.UpdateHistory(firstFrame);
            break;
        }
        case VIBE: {
            vibe  = ViBe(firstFrame.cols, firstFrame.rows);
            vibe.Initialize(firstFrame);
            break;
        }
        case MOG: {
            //doesn't work
            mog2 = cv::BackgroundSubtractorMOG2(3, 100, true);
            break;
        }
        case NONE : {
            return;
        }
    }
}

void BlobGetter::GetForegroundMap(cv::Mat input, cv::Mat& foregroundMap)
{
    switch (this->backgroundSubtractor){
        case TIMEDISPERSION: {
            if (timeDispersion.UpdateHistory(input)){
                timeDispersion.Process(foregroundMap);
            }
            break;
        }
        case VIBE: {
            vibe.Process(input, foregroundMap);
            break;
        }
        case MOG: {
            mog2(input, foregroundMap, 0);
        }
        case NONE : {
            foregroundMap = cv::Mat(input.size(), CV_8UC1);
        }
    }
}

void BlobGetter::GetSkinRegionMap(cv::Mat input, cv::Mat& skinMap)
{
    cv::GaussianBlur(input, input, cv::Size(5, 5), 1, 1);
    cv::cvtColor(input, skinMap, CV_BGR2YCrCb);
    if (isThresholdsAdapted)
        cv::inRange(skinMap, this->adaptLow, this->adaptHigh, skinMap);
    else
        cv::inRange(skinMap, this->defaultLow, this->defaultHigh, skinMap);
    cv::erode(skinMap, skinMap, cv::Mat());
    cv::dilate(skinMap, skinMap, cv::Mat());
}

void BlobGetter::MedianFilter(cv::Mat input, cv::Mat &output, size_t times)
{
    cv::Mat mask = input, md;
    for (size_t i = 0; i < times; ++i) {
        cv::filter2D(mask, md, CV_32F, this->medianKernel);
        mask = (md > 155);
    }
    output = mask;
}

void BlobGetter::AdaptColourThresholds(cv::Mat input, cv::Rect roi)
{
    cv::Mat ycrcb;
    ResetColourThresholds();
    cv::cvtColor(input, ycrcb, CV_BGR2YCrCb);
    ColorFinder finder = ColorFinder();
    finder.find(adaptLow, adaptHigh, 1, ycrcb);
    finder.find(adaptLow, adaptHigh, 2, ycrcb);
    this->isThresholdsAdapted = true;
}

void BlobGetter::ResetColourThresholds()
{
    this->adaptLow = cv::Scalar(20, 0, 0);
    this->adaptHigh = cv::Scalar(140, 0, 0);
    this->isThresholdsAdapted = false;
}
