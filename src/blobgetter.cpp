#include "blobgetter.h"

BlobGetter::BlobGetter(BackgroudSubstractionTechique backgroundSubtractor)
{
    this->backgroundSubtractor = backgroundSubtractor;
    this->isFirstFrame = true;
}

void BlobGetter::Process(cv::Mat input, cv::Mat &skinMap, cv::Mat &foregroundMap)
{
    //cv::blur(input, input, cv::Size(3, 3));
    if (isFirstFrame){
        skinMap = new cv::Mat(input.size(), CV_8UC1);
        foregroundMap = new cv::Mat(input.size(), CV_8UC1);
        InitializeBackgroundSubtractor(input);
        isFirstFrame = false;
    }
    else {
        GetForegroundMap(input, foregroundMap);
    }
    GetSkinRegionMap(input, skinMap);
}


void BlobGetter::InitializeBackgroundSubtractor(cv::Mat firstFrame)
{
    switch (this->backgroundSubtractor){
        case TIMEDISPERSION: {
            timeDispersion = TimeDispersion(firstFrame.cols, firstFrame.rows, 20, 20);
            timeDispersion.UpdateHistory(firstFrame);
            break;
        }
        case VIBE: {
            vibe  = ViBe(firstFrame.cols, firstFrame.rows);
            vibe.Initialize(firstFrame);
            break;
        }
    }
}

void BlobGetter::GetForegroundMap(cv::Mat input, cv::Mat* foregroundMap)
{
    switch (this->backgroundSubtractor){
        case TIMEDISPERSION: {
            if (timeDispersion.UpdateHistory(input)){
                timeDispersion.Process(*foregroundMap);
            }
            break;
        }
        case VIBE: {
            vibe.Process(input, *foregroundMap);
            break;
        }
    }
}

void BlobGetter::GetSkinRegionMap(cv::Mat input, cv::Mat* skinMap)
{

}
