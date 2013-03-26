#include "classifier.h"

Classifier::Classifier()
{
}

void Classifier::AddToTrainSet(std::vector<cv::Point> contour, int label)
{
    FeatureVector vector = getFeatures(contour);
    vector.label = label;
    trainVectors.push_back(vector);
}

void Classifier::Train()
{
    if (trainVectors.size() == 0) {
        std::cout << "Train vector is empty!" << std::endl;
    }
    assert(trainVectors.size() > 0);
    cv::Mat trainMat = cv::Mat(trainVectors.size(), trainVectors.at(0).VECTOR_SIZE, CV_32FC1);
    cv::Mat responsesMat = cv::Mat(trainVectors.size(), 1, CV_32SC1);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        responsesMat.at<int>(cv::Point(0, i)) = trainVectors.at(i).label;
        trainMat.at<float>(cv::Point(0, i)) = static_cast<float>(trainVectors.at(i).ecc);
        for (size_t m = 0; m < trainVectors.at(0).moments.size(); ++m)
            trainMat.at<float>(cv::Point(0, m + 1)) = static_cast<float>(trainVectors.at(i).moments.at(m));
        for (size_t h = 0; h < trainVectors.at(0).histogram.size(); ++h)
            trainMat.at<float>(cv::Point(0, h + 1 + trainVectors.at(0).moments.size())) =
                                                static_cast<float>(trainVectors.at(i).moments.at(h));
        #ifdef DEBUG
            std::cout << "row = " << i << std::endl;
            for (size_t v = 0; v < trainVectors.at(0).VECTOR_SIZE; ++v) {
                std::cout << trainMat.at<float>(cv::Point(i, v)) << " ";
            }
            std::cout << std::endl;
        #endif
    }
    classifier = CvNormalBayesClassifier();
    classifier.train(trainMat, responsesMat);
}

int Classifier::Recognize(std::vector<cv::Point> contour)
{
    FeatureVector vector = getFeatures(contour);
    cv::Mat predictMat = cv::Mat(1, vector.VECTOR_SIZE, CV_32FC1);
    predictMat.at<float>(cv::Point(0, 0)) = static_cast<float>(vector.ecc);
    for (size_t m = 0; m < vector.moments.size(); ++m)
        predictMat.at<float>(cv::Point(1 + m, 0)) = static_cast<float>(vector.moments.at(m));
    for (size_t h = 0; h < vector.histogram.size(); ++h)
        predictMat.at<float>(cv::Point(1 + vector.moments.size() + h, 0)) = static_cast<float>(vector.histogram.at(h));
    cv::Mat resultMat = cv::Mat(1, 1, CV_32SC1);
    classifier.predict(predictMat, &resultMat);
    return resultMat.at<int>(cv::Point(0, 0));
}

size_t Classifier::GetTrainSetSize()
{
    return trainVectors.size();
}


double Classifier::getEccentricity(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    double a20 = moments.mu20 / moments.m00;
    double a02 = moments.mu02 / moments.m00;
    double a11 = moments.mu11 / moments.m00;
    double l1 = (a20 + a02) / 2 + sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    double l2 = (a20 + a02) / 2 - sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    return 1 - l2 / l1;
}

FeatureVector Classifier::getFeatures(std::vector<cv::Point> contour)
{
    FeatureVector vector = FeatureVector();
    vector.ecc = getEccentricity(contour);
    cv::HuMoments(cv::moments(contour), vector.moments);
    cv::Rect size = cv::boundingRect(contour);
    cv::Mat buf = cv::Mat(size.size(), CV_8UC1);
    IntegralOrientationHistogram histogram = IntegralOrientationHistogram(16, INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL,
                                                                          buf, 100);
    histogram.calculate();
    for (size_t i = 0; i < histogram.getHistogram().size(); ++i) {
        vector.histogram.push_back(histogram.getHistogram().at(i).value);
    }
    vector.VECTOR_SIZE = 1 + vector.moments.size() + vector.histogram.size();
    return vector;
}
