#include "classifier.h"

cv::Mat trainMat, responsesMat;
CvKNearest knn;

Classifier::Classifier()
{
    trainVectors = std::vector<FeatureVector>();
    this->isTrained = false;

}

Classifier::Classifier(std::string model) {
    trainVectors = std::vector<FeatureVector>();
    cv::FileStorage fs(model, cv::FileStorage::READ);
    fs["train"] >> trainMat;
    fs["responses"] >> responsesMat;
    fs.release();
    Train(false);
    std::cout << "Loaded from " << model << " and trained" << std::endl;
    this->isTrained = true;
}

void Classifier::AddToTrainSet(int label, std::vector<std::vector<double> > histograms)
{
    FeatureVector fv = getFeatureVector(label, histograms);
    std::cout << "Added label = " << label << " total frames = " << histograms.size() << std::endl;
    std::cout << "a = " << fv.a.size() << " b = " << fv.b.size() << " c = " << fv.c.size() << " d = " << fv.d.size() << std::endl;
    trainVectors.push_back(fv);
}

FeatureVector Classifier::getFeatureVector(int label, std::vector<std::vector<double> > histograms)
{
    FeatureVector fv;
    fv.VECTOR_SIZE = 16 * 4;
    fv.label = label;
    fv.a = getMedianHistogram(histograms, 0, (histograms.size() - 1) / 4);
    fv.b = getMedianHistogram(histograms, (histograms.size() - 1) / 4, (histograms.size() - 1) / 2);
    fv.c = getMedianHistogram(histograms, (histograms.size() - 1) / 2, 3 * (histograms.size() - 1) / 4);
    fv.d = getMedianHistogram(histograms, 3 * (histograms.size() - 1) / 4, histograms.size() - 1);
    return fv;
}

std::vector<float> Classifier::getMedianHistogram(std::vector<std::vector<double> > histograms, int begin, int end)
{
    std::vector<float> result = std::vector<float>(histograms.at(0).size(), 0.0);
    for (int i = begin; i < end; ++i) {
        for (size_t a = 0; a < result.size(); ++a)
            result.at(a) += static_cast<float>(histograms.at(i).at(a));
    }
    for (size_t i = 0; i < result.size(); ++i) {
        result.at(i) /= end - begin;
    }
    return result;
}

void Classifier::Train(bool generateTrain)
{
    if (trainVectors.size() == 0 && generateTrain) {
        std::cout << "Train vector is empty!" << std::endl;
        return;
    }
    if (generateTrain) {
        trainMat = cv::Mat(trainVectors.size(), trainVectors.at(0).VECTOR_SIZE, CV_32FC1);
        responsesMat = cv::Mat(trainVectors.size(), 1, CV_32FC1);
        for (size_t i = 0; i < trainVectors.size(); ++i) {
            responsesMat.at<float>(i, 0) =  static_cast<float>(trainVectors.at(i).label);
            for (size_t a = 0; a < trainVectors.at(i).a.size(); ++a)
                trainMat.at<float>(i, a) = trainVectors.at(i).a.at(a);
            for (size_t b = 0; b < trainVectors.at(i).b.size(); ++b)
                trainMat.at<float>(i, b + 16) = trainVectors.at(i).b.at(b);
            for (size_t c = 0; c < trainVectors.at(i).c.size(); ++c)
                trainMat.at<float>(i, c + 32) = trainVectors.at(i).c.at(c);
            for (size_t d = 0; d < trainVectors.at(i).d.size(); ++d)
                trainMat.at<float>(i, d + 48) = trainVectors.at(i).d.at(d);
        }
    }

    knn = CvKNearest();
    knn.train(trainMat, responsesMat, cv::Mat(), false, 7);

    this->isTrained = true;
    if (generateTrain) {
        serializeTrainingVectors("vectors.csv");
        cv::FileStorage fs("data.xml", cv::FileStorage::WRITE);
        fs << "train" <<  trainMat;
        fs << "responses" << responsesMat;
        fs.release();
    }
}

bool Classifier::IsTrained()
{
    return this->isTrained;
}

int Classifier::Recognize(std::vector<std::vector<double> > histograms)
{
    clock_t start, stop;
    start = clock();
    FeatureVector vector = getFeatureVector(0, histograms);
    cv::Mat predictMat = cv::Mat(1, vector.VECTOR_SIZE, CV_32FC1);
    for (size_t a = 0; a < vector.a.size(); ++a)
        predictMat.at<float>(a) = vector.a.at(a);
    for (size_t b = 0; b < vector.b.size(); ++b)
        predictMat.at<float>(b + 16) = vector.b.at(b);
    for (size_t c = 0; c < vector.c.size(); ++c)
        predictMat.at<float>(c + 32) = vector.c.at(c);
    for (size_t d = 0; d < vector.d.size(); ++d)
        predictMat.at<float>(d + 48) = vector.d.at(d);
    cv::Mat distances, results, responses;
    int result = static_cast<int>(knn.find_nearest(predictMat, 6, results, responses, distances));
    std::cout << "ChiSqr = " << CalculateChiSqrDistances(vector, result) << std::endl;
    stop = clock();
    std::cout << "Recognized. Took " <<  stop - start << "  milliseconds" << std::endl;
    return result;
}

float calcMetrics(std::vector<float> input, std::vector<float> compare) {
    float sum = .0;
    for (size_t i = 0; i < input.size(); ++i) {
        sum += sqrt(pow(input.at(i) - compare.at(i), 2));
    }
    return sum;
}

float Classifier::CalculateChiSqrDistances(FeatureVector vector, int label) {
    float distances = 0;
    int precendents = 0;
    std::vector<float> input;
    input.insert(input.end(), vector.a.begin(), vector.a.end());
    input.insert(input.end(), vector.b.begin(), vector.b.end());
    input.insert(input.end(), vector.c.begin(), vector.c.end());
    input.insert(input.end(), vector.d.begin(), vector.d.end());
    for (int i = 0; i < responsesMat.rows; ++i) {
        size_t index = static_cast<size_t>(responsesMat.at<float>(i, 0)) - 1;
        if (label - 1 == index) {
            std::vector<float> compare;
            for (int j = 0; j < trainMat.cols; ++j) {
                compare.push_back(trainMat.at<float>(cv::Point(j, i)));
            }
            distances += calcMetrics(input, compare);
            precendents ++;
        }
    }
    return distances / precendents;
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
        for (size_t c = 0; c < current.c.size(); ++c)
            file << current.c.at(c) << ",";
        for (size_t d = 0; d < current.d.size(); ++d) {
            std::string appender = d == (current.d.size() - 1) ? "\n" : ",";
            file << current.d.at(d) << appender;
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
            std::string buf = "";
            std::vector<float> values;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] != ',' && i != line.size() - 1) {
                    buf += line[i];
                }
                else {
                    values.push_back(static_cast<float>(atof(buf.c_str())));
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
                for (size_t d = 0; d < 16; ++d)
                    newFeatureVector.d.push_back(values.at(d + 49));
                newFeatureVector.VECTOR_SIZE = 16 * 4;
                trainVectors.push_back(newFeatureVector);
            }
        }
    }
}

