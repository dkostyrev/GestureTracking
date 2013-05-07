#include "controller.h"
Controller::Controller()
{
    blobgetter = BlobGetter(TIMEDISPERSION);
    //classifier = Classifier();
    int i = open("model.xml", 0);
    if (i == -1) {
        close(i);
        std::cout << "model.xml doesn't exists, Training mode" << std::endl;
    } else {
        close(i);
        classifier = Classifier("model.xml");
    }
}

bool isGestureActive = false;
int threshold = 0;
int topthreshold = 0;
std::vector<cv::Mat> gesture;
MotionEstimator motionEstimator;
std::vector<std::vector<std::vector<double> > > saved;
std::vector<std::vector<std::vector<double> > > wrong;

void Controller::Process(cv::Mat frame)
{
    cv::Mat foregroundMat;
    blobgetter.getForegroundMap(frame, foregroundMat);
    if (foregroundMat.empty())
        return;

    if (threshold == 0) {
        threshold = static_cast<int>(frame.rows * frame.cols * 0.1);
        std::cout << "Threshold = " << threshold << std::endl;
    }
    if (topthreshold == 0) {
        topthreshold = static_cast<int>(frame.rows * frame.cols * 0.9);
        std::cout << "Top threshold = " << topthreshold << std::endl;
    }
    int count = cv::countNonZero(foregroundMat);
    if (count < topthreshold) {
        if (count > threshold && isGestureActive) {
            motionEstimator.AddFrame(foregroundMat);
            std::stringstream ss;
            ss << count;
            cv::putText(foregroundMat, "GESTURE", cv::Point(100, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0));
        }
        if (count > threshold && !isGestureActive) {
            isGestureActive = true;
            std::cout << "START" << std::endl;
            motionEstimator.clear();
        } else if (count < threshold && isGestureActive) {
            isGestureActive = false;
            std::cout << "STOP" << std::endl;
            if (motionEstimator.GetFrameCount() >= 5) {
                if (classifier.IsTrained())
                    classify();
                else
                    labelAndTrain();
            }
        }
    }
    cv::imshow("TimeDisp", foregroundMat);
    checkKeys();
}

//////////////////////////////////////
//       COMPARE FUNCTIONS          //
//////////////////////////////////////
/*
bool compare (std::vector<std::vector<double> > a, std::vector<std::vector<double> > b) {
    return a.size() < b.size();
}

std::vector<std::vector<std::vector<double> > > Controller::resizeToMax(std::vector<std::vector<std::vector<double> > > data, std::vector<std::vector<double> > max) {
    std::vector<std::vector<std::vector<double> > > currents;
    for (size_t s = 0; s < data.size(); ++s) {
        std::vector<std::vector<double> > current;
        double map = data.at(s).size() / max.size();
        for (size_t i = 0; i < max.size(); ++i)
            current.push_back(data.at(s).at(i * map));
        std::cout << current.size() << std::endl;
        currents.push_back(current);
    }
    return currents;
}

void Controller::getEuclidianDistance(std::vector<std::vector<std::vector<double> > > trainInput, std::vector<std::vector<std::vector<double> > > queryInput) {
    std::vector<std::vector<double> > max = *std::max_element(trainInput.begin(), trainInput.end(), compare);
    std::cout << "Max size is " << max.size() << std::endl;
    std::vector<std::vector<std::vector<double> > > train = resizeToMax(trainInput, max);
    std::vector<std::vector<std::vector<double> > > query = resizeToMax(queryInput, max);
    std::vector<std::vector<double> > median = getMedianVector(train);
    for (size_t i = 0; i < query.size(); ++i) {
        double sum = 0;
        for (size_t c = 0; c < query.at(i).size(); ++c) {
            for (size_t v = 0; v < query.at(i).at(c).size(); ++v) {
                sum += sqrt(pow(query.at(i).at(c).at(v) - median.at(c).at(v), 2));
            }
        }
        std::cout << "i = " << i << " to median = " << sum << std::endl;
    }

    for (size_t i = 0; i < query.size(); ++i) {
        double sum = 0;
        for (size_t c = 0; c < query.at(i).size(); ++c) {
            for (size_t v = 0; v < query.at(i).at(c).size(); ++v) {
                sum += sqrt(pow(query.at(i).at(c).at(v) - max.at(c).at(v), 2));
            }
        }
        std::cout << "i = " << i << " to max = " << sum << std::endl;
    }
}

std::vector<std::vector<double> > Controller::getMedianVector(std::vector<std::vector<std::vector<double> > > data) {
    std::vector<std::vector<double> > median = std::vector<std::vector<double> >(data.at(0).size(), std::vector<double>(16, 0));
    for (size_t i = 0; i < data.size(); ++i) {              //sample
        for (size_t c = 0; c < data.at(i).size(); ++c) {    //histogram
            for (size_t v = 0; v < data.at(i).at(c).size(); ++v) {
                median.at(c).at(v) += data.at(i).at(c).at(v);
            }
        }
    }

    for (size_t i = 0; i < median.size(); ++i) {
        for (size_t v = 0; v < 16; ++v)
            median.at(i).at(v) /= data.size();
    }
    return median;
}
*/
//////////////////////////////////////
//     END OF COMPARE FUNCTIONS     //
//////////////////////////////////////

void Controller::classify() {
    std::vector<std::vector<double> > histograms;
    motionEstimator.calculateMotionHistograms(histograms, false, false);
    int label = classifier.Recognize(histograms);
    std::cout << "Recognized as: = " << label << std::endl;
}

void Controller::labelAndTrain() {
    std::cout << "Label motion..." << std::endl;
    int key = cv::waitKey(0);
    int label = 0;
    if (key == 27) {
        std::cout << "skipping.." << std::endl;
        return;
    }
    switch ((char) key) {
        case '0':
            label = 0;
            break;
        case '1':
            label = 1;
            break;
        case '2':
            label = 2;
            break;
        case '3':
            label = 3;
            break;
        case '4':
            label = 4;
            break;
        case '5':
            label = 5;
            break;
    }
    //motionEstimator.ShowAllFrames();
    std::vector<std::vector<double> > histograms;

    motionEstimator.calculateMotionHistograms(histograms, false, false);
    classifier.AddToTrainSet(label, histograms);
    //if (label == 1)
    //    saved.push_back(histograms);
    //if (label == 0)
    //    wrong.push_back(histograms);
}

void Controller::checkKeys() {
    int key = cv::waitKey(1);
    switch ((char) key) {
    case 's':
        std::cout << "serializing..." << std::endl;
        classifier.serializeTrainingVectors("vectors.csv");
        break;
    case 't':
        std::cout << "training..." << std::endl;
        classifier.Train(true);
        //getEuclidianDistance(saved, saved);
        //getEuclidianDistance(saved, wrong);
    default:
        break;
    }
 }
