#include "blobgetter.h"
BlobGetter::BlobGetter(BackgroudSubtractionTechique backgroundSubtractor)
{
    this->backgroundSubtractor = backgroundSubtractor;
    isFirstFrame = true;
    medianKernel = cv::Mat(9, 9, CV_32F);
    medianKernel.setTo(cv::Scalar(1./81));
    defaultLow = cv::Scalar(20, 133, 80);
    defaultHigh = cv::Scalar(140, 173, 120);
    ResetColourThresholds();
}

void BlobGetter::Process(cv::Mat &input, cv::Mat &skinMap, cv::Mat &foregroundMap)
{
    cv::GaussianBlur(input, input, cv::Size(5, 5), 1, 1);
    cv::cvtColor(input, skinMap, CV_BGR2YCrCb);
    if (isThresholdsAdapted)
        cv::inRange(skinMap, this->adaptLow, this->adaptHigh, skinMap);
    else
        cv::inRange(skinMap, this->defaultLow, this->defaultHigh, skinMap);
    cv::erode(skinMap, skinMap, cv::Mat());
    cv::dilate(skinMap, skinMap, cv::Mat());
    //FilterBySize(skinMap, skinMap);
    MedianFilter(skinMap, skinMap, 1);
    ProcessForegroundMap(input, foregroundMap);
}

void BlobGetter::GetForegroundMap(cv::Mat &input, cv::Mat& output) {
    cv::cvtColor(input, input, CV_BGR2GRAY);
    if (isFirstFrame){
        output = cv::Mat(input.size(), CV_8UC1);
        InitializeBackgroundSubtractor(input);
        isFirstFrame = false;
    }
    else {
        ProcessForegroundMap(input, output);
    }
}

void BlobGetter::GetMixedMap(cv::Mat &input, cv::Mat& output) {
    cv::Mat skinMap, foregroundMap;
    Process(input, skinMap, foregroundMap);
    if (skinMap.empty() || foregroundMap.empty())
        return;
    cv::bitwise_and(foregroundMap, skinMap, output);
}

void BlobGetter::FilterBySize(cv::Mat &rawSkinMap, cv::Mat& skinMap)
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


void BlobGetter::InitializeBackgroundSubtractor(cv::Mat &firstFrame)
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
        case NO : {
            return;
        }
    }
}

void BlobGetter::MedianFilter(cv::Mat &input, cv::Mat &output, size_t times)
{
    cv::Mat mask = input, md;
    for (size_t i = 0; i < times; ++i) {
        cv::filter2D(mask, md, CV_32F, this->medianKernel);
        mask = (md > 155);
    }
    output = mask;
}

void BlobGetter::ProcessForegroundMap(cv::Mat &input, cv::Mat &output)
{
    switch (backgroundSubtractor){
        case TIMEDISPERSION: {
            if (timeDispersion.UpdateHistory(input)){
                timeDispersion.Process(output);
            }
            break;
        }
        case VIBE: {
            vibe.Process(input, output);
            break;
        }
        case MOG: {
            mog2(input, output, 0);
        }
        case NO : {
            output = cv::Mat(input.size(), CV_8UC1);
        }
    }
    if (!output.empty())
        MedianFilter(output, output, 2);
}

void BlobGetter::DispMap(cv::Mat &input, cv::Mat &output, int threshold)
{
    cv::cvtColor(input, input, CV_BGR2GRAY);
    input.convertTo(input, CV_32F);
    cv::Mat md, sqmd, mdsq, sq;
    cv::Mat k = (cv::Mat_<float>(3,3) << 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9);
    cv::filter2D(input, md, CV_32F, k);
    cv::pow(input, 2, sq);
    cv::pow(md, 2, sqmd);
    cv::filter2D(sq, mdsq, CV_32F, k);
    output= mdsq - sqmd;
    output = (output >= threshold * threshold);
    //MedianFilter(output, output, 1);
}

void BlobGetter::AdaptColourThresholds(cv::Mat &input, cv::Rect roi)
{
    cv::Mat ycrcb, crPlot, cbPlot;
    ResetColourThresholds();
    cv::cvtColor(input, ycrcb, CV_BGR2YCrCb);
    ycrcb = ycrcb(roi);
    cv::imshow("roi",ycrcb);
    ColorFinder finder = ColorFinder();
    finder.Find(adaptLow, adaptHigh, 1, ycrcb, 10);
    finder.Plot();
    crPlot = finder.GetPlottedMat();
    cv::imshow("crPlot", crPlot);
    finder.Find(adaptLow, adaptHigh, 2, ycrcb, 10);
    finder.Plot();
    cbPlot = finder.GetPlottedMat();
    cv::imshow("cbPlot", cbPlot);
    std::cout << adaptLow.val[0] << "< y  <" << adaptHigh[0] << std::endl;
    std::cout << adaptLow.val[1] << "< cr  <" << adaptHigh[1] << std::endl;
    std::cout << adaptLow.val[2] << "< cb  <" << adaptHigh[2] << std::endl;
    this->isThresholdsAdapted = true;
}

void BlobGetter::ResetColourThresholds()
{
    this->adaptLow = defaultLow;
    this->adaptHigh = defaultHigh;
    this->isThresholdsAdapted = false;
}
