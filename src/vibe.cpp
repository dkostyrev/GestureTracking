#include "vibe.h"

ViBe::ViBe()
{
}

ViBe::ViBe(int width, int height)
{
    this->width = width;
    this->height = height;
}

void ViBe::Initialize(cv::Mat firstframe)
{
    N = 2;
    R = 40;
    smin = 1;
    r_s = 20 ;
    samples = std::vector<cv::Mat>(N);
    for (size_t i = 0; i < samples.size(); ++i){
        samples[i] = firstframe;
    }
    rng = cv::RNG();
}

void ViBe::Process(cv::Mat image, cv::Mat& segMap)
{
    segMap = cv::Mat(image.size(), CV_8UC1);
    for (int x = 0; x < image.cols ; x++){
        for (int y = 0 ; y < image.rows; y++) {
            int count = 0, index = 0 , dist = 0;
            while (count < smin && index < N){
                dist = EuclidianDistance(image.at<uchar>(cv::Point(x,y)), samples.at(index).at<uchar>(cv::Point(x,y)));
                if (dist < R) {
                    count++;
                }
                index++;
            }
            if (count >= smin) {
                segMap.at<uchar>(cv::Point(x,y)) = (uchar) 0;
                int rand = rng.uniform(0, r_s-1);
                if (rand == 0 ) {
                    rand = rng.uniform(0, N-1);
                    samples.at(rand).at<uchar>(cv::Point(x,y)) = image.at<uchar>(cv::Point(x,y));
                }
                rand = rng.uniform(0, r_s-1);
                if (rand == 0) {
                    int xNG, yNG;
                    xNG = rng.uniform(0, image.cols-1);
                    yNG = rng.uniform(0, image.rows-1);
                    rand = rng.uniform(0, N-1);
                    samples.at(rand).at<uchar>(cv::Point(xNG, yNG)) = image.at<uchar>(cv::Point(xNG, yNG));
                }
            }
            else {
                segMap.at<uchar>(cv::Point(x,y)) = (uchar) 255;
            }
        }
    }
}

int ViBe::EuclidianDistance(uchar a, uchar b)
{
	return sqrt(std::pow(static_cast<float>(a - b), static_cast<float>(2)));
}
