#include "blobgetter.h"
BlobGetter::BlobGetter(BackgroudSubstractionTechique backgroundSubtractor)
{
    this->backgroundSubtractor = backgroundSubtractor;
    this->isFirstFrame = true;
    this->medianKernel = cv::Mat(9, 9, CV_32F);
    this->medianKernel.setTo(cv::Scalar(1./81));
}

void BlobGetter::Process(cv::Mat input, cv::Mat &skinMap, cv::Mat &foregroundMap)
{
    GetSkinRegionMap(input, skinMap);
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


void BlobGetter::InitializeBackgroundSubtractor(cv::Mat firstFrame)
{
    switch (this->backgroundSubtractor){
        case TIMEDISPERSION: {
            timeDispersion = TimeDispersion(firstFrame.cols, firstFrame.rows, 5, 7);
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
    cv::cvtColor(input, skinMap, CV_BGR2YCrCb);
    cv::inRange(skinMap, cv::Scalar(0, 133, 80), cv::Scalar(180, 173, 120), skinMap);
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
