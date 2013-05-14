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

size_t MotionEstimator::GetFrameCount()
{
    return frames.size();
}

void MotionEstimator::clear()
{
    frames.clear();
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


struct calculate_histograms_body {
  calculate_histograms_body(std::vector<std::vector<double> >* m_histograms, std::vector<cv::Mat> *m_frames,
                            bool m_plot, bool m_save, cv::Point m_histCenter)
  {
      frames = m_frames;
      histograms = m_histograms;
      save = m_save;
      plot = m_plot;
      histCenter = m_histCenter;
  }
  bool save, plot;
  cv::Point histCenter;
  std::vector<cv::Mat> *frames;
  std::vector<std::vector<double> >* histograms;

  void operator()( const tbb::blocked_range<size_t>& range ) const
  {
    for(size_t k = range.begin(); k < range.end(); k += 1 ) {
        BlobIntegralHistogram histogram = BlobIntegralHistogram(16, frames->at(k), histCenter);
        histogram.Calculate();
        if (plot)
            histogram.Plot();
        std::vector<double> currentHist;
        for (size_t h = 0; h < histogram.histogram.size(); ++h) {
            currentHist.push_back(histogram.histogram.at(h).value);
        }
        histograms->at(k) = currentHist;
        if (plot) {
            if (save) {
                std::stringstream str;
                str << k << "_hist.jpeg";
                cv::imwrite(str.str(), histogram.circularHistogram);
                str.str("");
                str << k << ".jpeg";
                cv::imwrite(str.str(), frames->at(k));
            }
            cv::imshow("hist", histogram.circularHistogram);
            cv::waitKey(0);
        }
    }
  }
};


void MotionEstimator::calculateMotionHistograms(std::vector<std::vector<double> >& histograms, bool plot, bool save)
{
    std::string dir = "";
    if (plot && save) {
        time_t t = time(0);
        struct tm * now = localtime(&t);
        std::stringstream str;
        str << now->tm_hour << "_" << now->tm_min << "_" << now->tm_sec << std::endl;
        dir = str.str();
        mkdir(dir.c_str());
    }
    BlobProcessor blobProcessor = BlobProcessor();
    int x = 0;
    int y = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::Point center = blobProcessor.getCenterOfMasses(frames.at(i));
        x += center.x;
        y += center.y;
    }
    cv::Point histCenter = cv::Point(static_cast<int>(x / frames.size()), static_cast<int>(y / frames.size()));
    std::cout << "Histogram center = " << histCenter << std::endl;
    histograms = std::vector<std::vector<double> >(frames.size(), std::vector<double>());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, frames.size()), calculate_histograms_body(&histograms, &frames, save, plot, histCenter));
    frames.clear();
}
