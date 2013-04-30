#include "classifier.h"

Classifier::Classifier()
{
    trainVectors = std::vector<FeatureVector>();
    this->isTrained = false;
}

Classifier::Classifier(std::string model) {
    classifier = CvANN_MLP();

    //deserializeTrainingVectors(model);
    //Train(false);
    classifier.load(model.c_str());
    std::cout << "Loaded perceptrone" << std::endl;
    std::cout << classifier.get_layer_count() << std::endl;
    this->isTrained = true;
}

void Classifier::AddToTrainSet(int label, std::vector<std::vector<double> > histograms)
{
    std::cout << "Added label = " << label << " total frames = " << histograms.size() << std::endl;
    FeatureVector fv;
    fv.VECTOR_SIZE = 16 * 3;
    fv.label = label;
    std::cout << "Will use 0, " << static_cast<int>(histograms.size() / 2) << ", " << histograms.size() - 1;
    fv.a = histograms.at(0);
    fv.b = histograms.at(static_cast<int>(histograms.size() / 2));
    fv.c = histograms.at(histograms.size() - 1);
    trainVectors.push_back(fv);
}

void Classifier::Train(bool save)
{
    serializeTrainingVectors("vectors.csv");
    if (trainVectors.size() == 0) {
        std::cout << "Train vector is empty!" << std::endl;
        return;
    }
    cv::Mat trainMat = cv::Mat(trainVectors.size(), trainVectors.at(0).VECTOR_SIZE, CV_32FC1);
    cv::Mat responsesMat = cv::Mat(trainVectors.size(), 1, CV_32FC1);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        responsesMat.at<float>(cv::Point(0, i)) =  static_cast<float>(trainVectors.at(i).label);
        for (size_t a = 0; a < trainVectors.at(i).a.size(); ++a)
            trainMat.at<float>(cv::Point(a, i)) = static_cast<float>(trainVectors.at(i).a.at(a));
        for (size_t b = 0; b < trainVectors.at(i).b.size(); ++b)
            trainMat.at<float>(cv::Point(b + 16, i)) = static_cast<float>(trainVectors.at(i).b.at(b));
        for (size_t c = 0; c < trainVectors.at(i).c.size(); ++c)
            trainMat.at<float>(cv::Point(c + 32, i)) = static_cast<float>(trainVectors.at(i).c.at(c));
        #ifdef DEBUG
            std::cout << "row = " << i << std::endl;
            for (int v = 0; v < trainVectors.at(0).VECTOR_SIZE; ++v) {
                std::cout << trainMat.at<float>(cv::Point(v, i)) << " ";
            }
            std::cout << std::endl;
        #endif
    }
    //classifier = cv::NormalBayesClassifier();
    classifier = CvANN_MLP();
    cv::Mat size = cv::Mat(1, 3, CV_32SC1);
    size.at<int>(0,0) = trainVectors.at(0).VECTOR_SIZE;
    size.at<int>(0,1) = 72;
    size.at<int>(0,2) = 1;
    classifier.create(size);
    classifier.train(trainMat, responsesMat, cv::Mat());
    this->isTrained = true;
    if (save) {
        classifier.save("model.xml");
        serializeTrainingVectors("vectors.csv");
    }
}

bool Classifier::IsTrained()
{
    return this->isTrained;
}

int Classifier::Recognize(std::vector<std::vector<double> > histograms)
{
    FeatureVector vector;
    vector.a = histograms.at(0);
    vector.b = histograms.at(static_cast<int>(histograms.size() / 2));
    vector.c = histograms.at(histograms.size() - 1);
    vector.label = 0;
    vector.VECTOR_SIZE = 16 * 3;
    cv::Mat predictMat = cv::Mat(1, vector.VECTOR_SIZE, CV_32FC1);
    for (size_t a = 0; a < vector.a.size(); ++a)
        predictMat.at<float>(cv::Point(a, 0)) = static_cast<float>(vector.a.at(a));
    for (size_t b = 0; b < vector.b.size(); ++b)
        predictMat.at<float>(cv::Point(b + 16, 0)) = static_cast<float>(vector.b.at(b));
    for (size_t c = 0; c < vector.c.size(); ++c)
        predictMat.at<float>(cv::Point(c + 32, 0)) = static_cast<float>(vector.c.at(c));
    cv::Mat resultMat = cv::Mat(1, 1, CV_32FC1);
    classifier.predict(predictMat, resultMat);
    return resultMat.at<float>(cv::Point(0, 0));
}

size_t Classifier::GetTrainSetSize()
{
    return trainVectors.size();
}

void Classifier::serializeTrainingVectors(std::string filename)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios_base::app);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        FeatureVector current = trainVectors.at(i);
        file << current.label << ",";
        for (size_t a = 0; a < current.a.size(); ++a)
            file << current.a.at(a) << ",";
        for (size_t b = 0; b < current.b.size(); ++b)
            file << current.b.at(b) << ",";
        for (size_t c = 0; c < current.c.size(); ++c) {
            std::string appender = c == (current.c.size() - 1) ? "\n" : ",";
            file << current.c.at(c) << appender;
        }
    }
    file.flush();
    file.close();
}

/*
void Classifier::serializeTrainingVectors(std::string filename)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios_base::app);
    for (size_t i = 0; i < trainVectors.size(); ++i) {
        FeatureVector current = trainVectors.at(i);
        file << current.label << ",";
        file << current.histograms.size() << ",";
        for (size_t a = 0; a < current.histograms.size(); ++a) {
            for (size_t h = 0; h < current.histograms.at(a).size(); ++h) {
                std::string appender = ",";
                if (a == current.histograms.size() - 1 && h == current.histograms.at(a).size() - 1)
                    appender = "\n";
                std::cout << "appender = " << appender << std::endl;
                file << current.histograms.at(a).at(h) << appender;
            }
        }
    }
    file.flush();
    file.close();
}
*/

void Classifier::deserializeTrainingVectors(std::string filename)
{
    std::ifstream file (filename.c_str());
    std::string line;
    if (file.is_open()) {
        while (file.good()) {
            std::getline(file, line);
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
                for (size_t a = 0; a < 16; ++a)
                    newFeatureVector.a.push_back(values.at(a + 1));
                for (size_t b = 0; b < 16; ++b)
                    newFeatureVector.b.push_back(values.at(b + 17));
                for (size_t c = 0; c < 16; ++c)
                    newFeatureVector.c.push_back(values.at(c + 33));
                trainVectors.push_back(newFeatureVector);
            }
        }
    }
}

