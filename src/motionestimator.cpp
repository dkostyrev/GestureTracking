#include "motionestimator.h"

MotionEstimator::MotionEstimator()
{
    this->frames = std::vector<cv::Mat>();
}

void MotionEstimator::AddFrame(cv::Mat frame)
{
    cv::Mat buf;
    frame.copyTo(buf);
    this->frames.push_back(buf);
}

void MotionEstimator::ShowAllFrames() {
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::imshow("Saved frame", frames.at(i));
        cv::waitKey(100);
    }
}

void MotionEstimator::GetMotionMat(cv::Mat &result)
{
    if (frames.size() > 1) {
        result = cv::Mat(frames.at(0).size(), CV_8UC1);
        result.setTo(cv::Scalar(0));
        for (size_t i = 0; i < frames.size(); ++i) {
            result += frames.at(i).setTo(cv::Scalar(255 / (frames.size() - i)), frames.at(i));;
        }
        std::cout << "frames = " << frames.size() << std::endl;
        cv::imshow("result", result);
        //getMotionVector(result, frames.size());
        frames.clear();
    }
}

void MotionEstimator::getMotionVector(cv::Mat result, size_t framesSize)
{
    cv::Mat bufResult;
    result.copyTo(bufResult);
    cv::Mat buf;
    for (size_t i = 0; i < framesSize; ++i) {
        cv::threshold(bufResult, buf, 255 / (framesSize - i), 255, CV_THRESH_TOZERO_INV);
        bufResult -= buf;
        cv::imshow("buf", buf);
        cv::waitKey(0);
        buf.setTo(cv::Scalar(0));
    }

}


void MotionEstimator::calculateMotionHistograms()
{
    for (size_t i = 0; i < frames.size(); ++i) {
        BlobIntegralHistogram histogram = BlobIntegralHistogram(16, frames.at(i));
        histogram.Calculate();
        histogram.Plot();
        cv::Mat hist = histogram.getPlottedMat();
        for (size_t h = 0; h < histogram.getHistogram().size(); ++h) {
            std::cout << histogram.getHistogram().at(h).value << ", ";
        }
        std::cout << std::endl;
        cv::imshow("hist", hist);
        cv::waitKey(0);
    }
}
