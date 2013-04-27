#include "classifier.h"

Classifier::Classifier()
{
    trainVectors = std::vector<FeatureVector>();
    isLoadedModel = false;
}

Classifier::Classifier(std::string model) {
    classifier = CvANN_MLP();

    deserializeTrainingVectors(model);
    Train(false);
    //classifier.load(model.c_str());
    std::cout << "Loaded perceptrone" << std::endl;
    std::cout << classifier.get_layer_count() << std::endl;
    isLoadedModel = true;
}

void Classifier::AddToTrainSet(cv::Size matSize, std::vector<cv::Point> contour, int label)
{
    FeatureVector vector = getFeatures(contour, matSize);
    vector.label = label;
    trainVectors.push_back(vector);
    std::cout << "Added" << "Total size = " << trainVectors.size() << std::endl;

}

void Classifier::Train(bool save)
{
    if (trainVectors.size() == 0) {
        std::cout << "Train vector is empty!" << std::endl;
    }
    cv::Mat trainMat = cv::Mat(trainVectors.size(), trainVectors.at(0).VECTOR_SIZE, CV_32FC1);
    cv::Mat responsesMat = cv::Mat(trainVectors.size(), 1, CV_32FC1);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        responsesMat.at<float>(cv::Point(0, i)) =  static_cast<float>(trainVectors.at(i).label);
        trainMat.at<float>(cv::Point(0, i)) = static_cast<float>(trainVectors.at(i).ecc);
        for (size_t m = 0; m < trainVectors.at(0).moments.size(); ++m)
            trainMat.at<float>(cv::Point(m + 1, i)) = static_cast<float>(trainVectors.at(i).moments.at(m));
        for (size_t h = 0; h < trainVectors.at(0).histogram.size(); ++h)
            trainMat.at<float>(cv::Point(1 + trainVectors.at(0).moments.size() + h, i)) =
                                                static_cast<float>(trainVectors.at(i).histogram.at(h));
        #ifdef DEBUG
            std::cout << "row = " << i << std::endl;
            for (size_t v = 0; v < trainVectors.at(0).VECTOR_SIZE; ++v) {
                std::cout << trainMat.at<float>(cv::Point(v, i)) << " ";
            }
            std::cout << std::endl;
        #endif
    }
    //classifier = cv::NormalBayesClassifier();
    classifier = CvANN_MLP();
    cv::Mat size = cv::Mat(1, 3, CV_32SC1);
    size.at<int>(0,0) = trainVectors.at(0).VECTOR_SIZE;
    size.at<int>(0,1) = 32;
    size.at<int>(0,2) = 1;
    classifier.create(size);
    classifier.train(trainMat, responsesMat, cv::Mat());
    this->isLoadedModel = true;
    if (save) {
        classifier.save("model.xml");
        serializeTrainingVectors("vectors.csv");
    }
}

int Classifier::Recognize(std::vector<cv::Point> contour, cv::Size matSize)
{
    FeatureVector vector = getFeatures(contour, matSize);
	vector.label = 0;
    cv::Mat predictMat = cv::Mat(1, vector.VECTOR_SIZE, CV_32FC1);
    predictMat.at<float>(0) = static_cast<float>(vector.ecc);
    for (size_t m = 0; m < vector.moments.size(); ++m)
        predictMat.at<float>(1 + m) = static_cast<float>(vector.moments.at(m));
    for (size_t h = 0; h < vector.histogram.size(); ++h)
        predictMat.at<float>(1 + vector.moments.size() + h) = static_cast<float>(vector.histogram.at(h));
		std::cout << "row = " << 0 << std::endl;
            for (size_t v = 0; v < vector.VECTOR_SIZE; ++v) {
                std::cout << predictMat.at<float>(cv::Point(v, 0)) << " ";
            }
            std::cout << std::endl;
    cv::Mat resultMat = cv::Mat(1, 1, CV_32FC1);
    classifier.predict(predictMat, resultMat);
    return resultMat.at<float>(cv::Point(0, 0));
}

bool Classifier::IsLoadedModel()
{
    return isLoadedModel;
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

FeatureVector Classifier::getFeatures(std::vector<cv::Point> contour, cv::Size matSize)
{
    FeatureVector vector = FeatureVector();
    vector.ecc = getEccentricity(contour);
    cv::HuMoments(cv::moments(contour), vector.moments);
    cv::Mat buf = cv::Mat(matSize, CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    contours.push_back(contour);
    cv::drawContours(buf, contours, 0, cv::Scalar(255), -1);
    IntegralOrientationHistogram histogram = IntegralOrientationHistogram(16, INTEGRAL_ORIENTATION_HISTOGRAM_SOBEL,
                                                                          buf, cv::boundingRect(contour), 100);
    histogram.calculate();
    for (size_t i = 0; i < histogram.getHistogram().size(); ++i) {
        vector.histogram.push_back(histogram.getHistogram().at(i).value);
    }
    vector.VECTOR_SIZE = 1 + vector.moments.size() + vector.histogram.size();
    return vector;
}

void Classifier::serializeTrainingVectors(std::string filename)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios_base::app);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        FeatureVector current = trainVectors.at(i);
        file << current.label << ",";
        file << current.ecc << ",";
        for (size_t h = 0; h < current.histogram.size(); ++h)
            file << current.histogram.at(h) << ",";
        for (size_t m = 0; m < current.moments.size(); ++m) {
            std::string appender = m != current.moments.size() - 1 ? "," : "\n";
            file << current.moments.at(m) << appender;
        }
    }
    file.flush();
    file.close();
}

void Classifier::deserializeTrainingVectors(std::string filename)
{
    std::ifstream file (filename.c_str());
    std::string line;
    if (file.is_open()) {
        while (file.good()) {
            std::getline(file, line);
            std::vector<cv::Point> contour;
            std::string buf = "";
            std::vector<double> values;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] != ',' && line[i] != '\n') {
                    buf += line[i];
                }
                else {
                    values.push_back(atof(buf.c_str()));
                    buf = "";
                }
            }
            std::cout << values.size() << std::endl;
            if (values.size() > 0) {
                FeatureVector newFeatureVector = FeatureVector();
                newFeatureVector.label = static_cast<int>(values.at(0));
                newFeatureVector.ecc = values.at(1);
                for (size_t i = 0; i < 16; ++i) {
                    newFeatureVector.histogram.push_back(values.at(i + 2));
                }
                for (size_t i = 0; i < 6; ++i) {
                    newFeatureVector.moments.push_back(values.at(i + 18));
                }
                newFeatureVector.VECTOR_SIZE = 1 + newFeatureVector.moments.size() + newFeatureVector.histogram.size();
                trainVectors.push_back(newFeatureVector);
            }
        }
    }
}

