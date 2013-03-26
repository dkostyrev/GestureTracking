#include "blobprocessor.h"

BlobProcessor::BlobProcessor()
{
    deserializeContours("open_palm.txt", trainContours);
    classifier = Classifier();
    trainInProgress = false;
    this->medianKernel = cv::Mat(9, 9, CV_32F);
    this->medianKernel.setTo(cv::Scalar(1./81));

}
std::vector<cv::Point> trainContours;
void BlobProcessor::Process(cv::Mat input, cv::Mat skinMap, cv::Mat foregroundMap)
{
    cv::Mat dispMap, buf = cv::Mat(input.size(), CV_8UC3);
    buf.setTo(cv::Scalar(0,0,0));
    std::vector<cv::Rect> rois;

    cv::cvtColor(input, dispMap, CV_BGR2GRAY);
    DispMap(dispMap, dispMap, 4);
    MedianFilter(dispMap, dispMap, 1);
    std::vector<std::vector<cv::Point> > dispContours, goodContours;
    getContours(dispMap, dispContours, CV_RETR_LIST);
    for (size_t i = 0; i < dispContours.size(); ++i) {
        if (cv::contourArea(dispContours.at(i)) > 150) {
            goodContours.push_back(dispContours.at(i));
        }
    }
    dispMap.setTo(cv::Scalar(0));
    cv::drawContours(dispMap, goodContours, -1, cv::Scalar(255));

    getCuttedRoisFromMap(skinMap, rois, 10);
    std::vector<std::vector<cv::Point> > contours;
    int key = cv::waitKey(1);
    if (key == (int) 't') {
        trainInProgress = true;
        trainProcedure(skinMap, rois);
    }
    for (size_t i = 0; i < rois.size(); ++i) {
        std::vector<cv::Point> contour = getMaxContourFromRoi(skinMap, rois.at(i));
        cv::rectangle(buf, rois.at(i), cv::Scalar(255, 0, 0));
        contours.clear();
        contours.push_back(contour);
        cv::drawContours(buf, contours, 0, cv::Scalar(255, 0, 0));
        if (classifier.GetTrainSetSize() > 0 && !trainInProgress) {
            int label = classifier.Recognize(contour);
            if (label > 0) {
                cv::rectangle(buf, rois.at(i), cv::Scalar(0, 255, 0));
            }
        }
    }
    //cv::imshow("result contours", buf);
    //cv::imshow("time disp", foregroundMap);
    cv::imshow("disp", dispMap);
}

void BlobProcessor::trainProcedure(cv::Mat map, std::vector<cv::Rect> rois) {
    std::vector<cv::Point> contour;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat buf = cv::Mat(map.size(), CV_8UC3);
    while (true) {
        for (size_t i = 0; i < rois.size(); ++i) {
            contour = getMaxContourFromRoi(map, rois.at(i));
            contours.clear();
            contours.push_back(contour);
            buf.setTo(cv::Scalar(0, 0, 0));
            cv::rectangle(buf, rois.at(i), cv::Scalar(255, 0, 0));
            cv::drawContours(buf, contours, 0, cv::Scalar(255, 0, 0));
            cv::imshow("classify contour", buf);
            int key = cv::waitKey(0);
            switch (key) {
                case int('0'):
                    std::cout << "classified as 0 class" << std::endl;
                    classifier.AddToTrainSet(contour, 0);
                    break;
                case int('1'):
                    std::cout << "classified as 1 class" << std::endl;
                    classifier.AddToTrainSet(contour, 1);
                    break;
                case int('2'):
                    std::cout << "classified as 2 class" << std::endl;
                    classifier.AddToTrainSet(contour, 2);
                    break;
                case int('3'):
                    std::cout << "classified as 3 class" << std::endl;
                    classifier.AddToTrainSet(contour, 3);
                    break;
                case int ('e'):
                    std::cout << "end of classification" << std::endl;
                    classifier.Train();
                    trainInProgress = false;
                    return;
                    break;
                case 27:
                    return;
                default:
                    break;
            }
        }
    }
}

std::vector<cv::Point> BlobProcessor::getMaxContourFromRoi(cv::Mat map, cv::Rect roi) {
    std::vector<std::vector<cv::Point> > contours;
    size_t maxI = -1; int maxArea = 0;
    getContours(map(roi), contours, CV_RETR_EXTERNAL, roi.tl());
    for (size_t c = 0; c < contours.size(); ++c) {
        if (cv::contourArea(contours.at(c)) > maxArea) {
            maxArea = cv::contourArea(contours.at(c));
            maxI = c;
        }
    }
    return contours.at(maxI);
}

void BlobProcessor::getCuttedRoisFromMap(cv::Mat map, std::vector<cv::Rect> & rois, int extra) {
    std::vector<std::vector<cv::Point> > contours;
    getContours(map, contours);
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours.at(i)) > 50) {
            std::vector<cv::Rect> rects;
            cutRoi(contours.at(i), rects);
            for (size_t r = 0; r < rects.size(); ++r) {
                if (rects.at(r).area() > 100) {
                    rects.at(r).x -= extra / 2;
                    rects.at(r).width += extra;
                    rects.at(r).y -= extra / 2;
                    rects.at(r).height += extra;
                    if (rects.at(r).x < 0)
                        rects.at(r).x = 0;
                    if (rects.at(r).width + rects.at(r).x >= map.cols)
                        rects.at(r).width = map.cols - rects.at(r).x;
                    if (rects.at(r).y < 0)
                        rects.at(r).y = 0;
                    if (rects.at(r).height + rects.at(r).y >= map.rows)
                        rects.at(r).height = map.rows - rects.at(r).y;

                    rois.push_back(rects.at(r));
                }
            }
        }
    }
}


void BlobProcessor::deserializeContours(std::string filename, std::vector<std::vector<cv::Point> > &contours)
{
    std::ifstream file (filename.c_str());
    std::string line;
    if (file.is_open()) {
        while (file.good()) {
            std::getline(file, line);
            std::vector<cv::Point> contour;
            std::string buf = "";
            cv::Point point = cv::Point2i(0, 0);
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] != ':' && line[i] != ';' && line[i] != ',')
                    buf += line[i];
                else if (line[i] == ';')
                    contours.push_back(contour);
                else if (line[i] == ':') {
                    point.x = atoi(buf.c_str());
                    buf = "";
                }
                else if (line[i] == ',') {
                    point.y = atoi(buf.c_str());
                    buf = "";
                    contour.push_back(point);
                }
            }
        }
    }
    file.close();
}


double euclidianDistance(cv::Point a, cv::Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int PREDICATE_PARAM = 0;
bool PredicateFunction2(cv::Point a, cv::Point b){
    if (euclidianDistance(a, b) < PREDICATE_PARAM)
        return true;
    else
        return false;
}

void BlobProcessor::cutRoi(std::vector<cv::Point> contour, std::vector<cv::Rect>& clusters) {
    std::vector<cv::Point> hull;
    PREDICATE_PARAM = 10;
    cv::convexHull(contour, hull);
    std::vector<int> labels;
    int totalLabels = cv::partition(hull, labels, PredicateFunction2);
    std::vector<cv::Rect> smallClusters = getRectsFromLabeledPoints(totalLabels, labels, hull);
    clusters = smallClusters;
    int med = 0;
    int total = 0;
    for (size_t i = 0; i < smallClusters.size(); ++i) {
        cv::Point c1 = (smallClusters.at(i).br() + smallClusters.at(i).tl()) * 0.5;
        for (size_t j = 0; j < smallClusters.size(); ++j) {
            if (i != j) {
                cv::Point c2 = (smallClusters.at(j).br() + smallClusters.at(j).tl()) * 0.5;
                med += static_cast<int> (euclidianDistance(c1, c2));
                total++;
            }
        }
    }
    if (total == 0)
        return;
    PREDICATE_PARAM = med / total;
    totalLabels = cv::partition(hull, labels, PredicateFunction2);
    clusters = getRectsFromLabeledPoints(totalLabels, labels, hull);
}

std::vector<cv::Rect> BlobProcessor::getRectsFromLabeledPoints(int totalLabels, std::vector<int> labels, std::vector<cv::Point> points) {
    std::vector<std::vector<cv::Point> > clustersPoints = std::vector<std::vector<cv::Point> >(totalLabels);
    std::vector<cv::Rect> clusters;
    for (size_t i = 0; i < labels.size(); ++i) {
        clustersPoints.at(labels.at(i)).push_back(points.at(i));
    }
    for (size_t i = 0; i < clustersPoints.size(); ++i) {
        clusters.push_back(cv::boundingRect(clustersPoints.at(i)));
    }
    return clusters;
}

void BlobProcessor::serializeContour(std::string filename, std::vector<cv::Point> contour)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios_base::app);
    for (size_t i = 0; i < contour.size(); ++i) {
        file << contour.at(i).x << "," << contour.at(i).y << "\n";
        /*if (i == contour.size() - 1)
            file << ";\n";
        else
            file << ",";
            */
    }
    file.flush();
    file.close();
}

cv::Point BlobProcessor::getCenterOfMasses(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    return cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

void BlobProcessor::getContours(cv::Mat input, std::vector<std::vector<cv::Point> > &contours, int method ,cv::Point offset) {
    cv::Mat buf;
    input.copyTo(buf);
    cv::findContours(buf, contours, method, CV_CHAIN_APPROX_SIMPLE, offset);
}

void BlobProcessor::getRects(cv::Mat input, std::vector<cv::Rect>& rects) {
    std::vector<std::vector<cv::Point> > contours = std::vector<std::vector<cv::Point> >();
    cv::Mat buf;
    input.copyTo(buf);
    cv::findContours(buf, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); ++i) {
        rects.push_back(cv::boundingRect(contours.at(i)));
    }
}

void BlobProcessor::MedianFilter(cv::Mat input, cv::Mat &output, size_t times)
{
    cv::Mat mask = input, md;
    for (size_t i = 0; i < times; ++i) {
        cv::filter2D(mask, md, CV_32F, this->medianKernel);
        mask = (md > 100);
    }
    output = mask;
}


void BlobProcessor::DispMap(cv::Mat input, cv::Mat &output, int threshold)
{
    input.convertTo(input, CV_32F);
    cv::Mat md, sqmd, mdsq, sq;
    cv::Mat k = (cv::Mat_<float>(3,3) << 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9);
    cv::filter2D(input, md, CV_32F, k);
    cv::pow(input, 2, sq);
    cv::pow(md, 2, sqmd);
    cv::filter2D(sq, mdsq, CV_32F, k);
    output= mdsq - sqmd;
    output = (output >= threshold * threshold);
}


