#include "blobprocessor.h"

BlobProcessor::BlobProcessor()
{
}

void BlobProcessor::Process(cv::Mat skinMap, cv::Mat input)
{
    cv::cvtColor(input, hsv, CV_BGR2HSV);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(skinMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> clusters;
    for (size_t i = 0; i < contours.size(); ++i) {
        clusters.push_back(cv::boundingRect(contours.at(i)));
    }

}

void BlobProcessor::SuppressBadClusters(cv::Mat input, std::vector<cv::Rect> cluster) {
    ColorFinder finder = ColorFinder();
    cv::Mat roi = hsv(cluster);
    std::cout << finder.getIntervalsCount(roi, 90) << std::endl;

}


