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
    std::vector<cv::Rect> goodClusters;
    GetGoodClusters(clusters, goodClusters);
    RecognizeClusters(input, goodClusters);
    for (size_t i = 0; i < goodClusters.size(); ++i) {
        cv::rectangle(input, goodClusters.at(i), cv::Scalar(255, 0, 0));
    }
    cv::imshow("goodclusters", input);
}

bool compareFunction(std::pair<int, int> a, std::pair<int, int> b) { return a.second < b.second; }

void BlobProcessor::GetGoodClusters(std::vector<cv::Rect> clusters, std::vector<cv::Rect>& goodClusters) {
    float PERCENTAGE = 0.4;
    std::vector<std::pair<int, int> > clustersValues;
    ColorFinder finder = ColorFinder();
    for (size_t i = 0; i < clusters.size(); ++i) {
        cv::Mat roi = hsv(clusters.at(i));
        std::pair<int, int> pair;
        pair.first = i;
        pair.second = finder.getIntervalsCount(roi, 90);
        clustersValues.push_back(pair);
    }
    std::sort(clustersValues.begin(), clustersValues.end(), compareFunction);
    int goodClustersCount = clusters.size() * PERCENTAGE;
    int currentGoodClustersValue = 0;
    for (size_t i = 0; i < goodClustersCount; ++i) {
        currentGoodClustersValue += clustersValues.at(clustersValues.size() - i - 1).second;
    }
    currentGoodClustersValue /= goodClustersCount;
    for (size_t i = clustersValues.size() - 1; i > 0; --i) {
        if (clustersValues.at(i).second >= currentGoodClustersValue) {
            goodClusters.push_back(clusters.at(clustersValues.at(i).first));
        }
    }

}

void BlobProcessor::RecognizeClusters(cv::Mat input, std::vector<cv::Rect> clusters) {
    //std::vector<cv::Mat> splitted;
    //cv::split(hsv, splitted);
    cv::cvtColor(input, input, CV_BGR2GRAY);
    cv::Mat output;
    DispMap(input, output);
    cv::imshow("contour", output);
    cv::waitKey();
    /*for (size_t i = 0; i < clusters.size(); ++i) {
        cv::Mat output;
        DispMap(input(clusters.at(i)), output);
        cv::imshow("contour", output);
        cv::waitKey();
    }*/
}


void BlobProcessor::DispMap(cv::Mat input, cv::Mat &output)
{
    input.convertTo(input, CV_32F);
    cv::Mat md, sqmd, mdsq, sq;
    cv::Mat k = (cv::Mat_<float>(3,3) << 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9);
    cv::filter2D(input, md, CV_32F, k);
    cv::pow(input, 2, sq);
    cv::pow(md, 2, sqmd);
    cv::filter2D(sq, mdsq, CV_32F, k);
    output= mdsq - sqmd;
    output = (output >= 5*5);
}


