#include "blobprocessor.h"

BlobProcessor::BlobProcessor() {}

/*
 * Copyright by  Nikolas Markou
 * original C# version from:
 * http://nmarkou.blogspot.ru/2012/03/contour-refinement.html
 */

void BlobProcessor::contourRefine(std::vector<cv::Point> contour, cv::Mat blobMask, std::vector<cv::Point>& refinedContour) {
    int normalOffset = 5;
    float featureThreshold = 10e100;
    float inertiaCoeff = 1.1;
    float multiplierCoeff = -1;
    std::vector<cv::Point> points;
    for (size_t i = 0; i < contour.size(); ++i) {
        int totalPoints = contour.size(),
            ki = (i + 1) % totalPoints,
            ik = (i >= 1) ? (i - 1) : (totalPoints - 1 + 1) % totalPoints;
        cv::Point current = contour.at(i),
                  next = contour.at(ki),
                  prev = contour.at(ik);
        cv::Point2f normalOut = normalAtPoint(prev, current, next, false),
                    normalIn = normalAtPoint(prev, current, next, true);
        cv::Point out = cv::Point(static_cast<int>(floor(normalOut.x * normalOffset)) + current.x,
                                  static_cast<int>(floor(normalOut.y * normalOffset)) + current.y),
                  in = cv::Point(static_cast<int>(floor(normalIn.x * normalOffset)) + current.x,
                                  static_cast<int>(floor(normalIn.y * normalOffset)) + current.y);
        std::vector<uchar> sampleIn = sampleLine(blobMask, current, in);
        std::vector<uchar> sampleOut = sampleLine(blobMask, current, out);
        float max = 0, sample = 0;
        int j = 0;
        bool inOut = true;
        for (size_t i = 0; i < sampleOut.size(); ++i) {
            sample = static_cast<float>(sampleOut.at(i)) + multiplierCoeff * static_cast<float>(pow(inertiaCoeff, static_cast<float>(i)));
            if (sample > max) {
                max = sample;
                j = i;
                inOut = false;
            }
        }
        std::cout << max << std::endl;
        for (size_t i = 0; i < sampleIn.size(); ++i) {
            sample = static_cast<float>(sampleIn.at(i)) + multiplierCoeff * static_cast<float>(pow(inertiaCoeff, static_cast<float>(i)));
            if (sample > max) {
                max = sample;
                j = i;
                inOut = true;
            }
        }
        if (max >= featureThreshold) {
            int x,y;
            double length, xlength, ylength;
            if (!inOut) {
                xlength = current.x - out.x;
                ylength = current.y - out.y;
                //length = sqrt(pow(xlength, 2) + pow(ylength, 2));
                x = static_cast<int>(floor(static_cast<float>(j) / sampleOut.size() * out.x * normalOffset));
                y = static_cast<int>(floor(static_cast<float>(j) / sampleOut.size() * out.y * normalOffset));
            } else {
                xlength = current.x - in.x;
                ylength = current.y - in.y;
                //length = sqrt(pow(xlength, 2) + pow(ylength, 2));
                x = static_cast<int>(floor(static_cast<float>(j) / sampleIn.size() * in.x * normalOffset));
                y = static_cast<int>(floor(static_cast<float>(j) / sampleIn.size() * in.y * normalOffset));
            }
            points.push_back(cv::Point(current.x + x, current.y + y));
        }
    }
    refinedContour.clear();
    std::cout << "points" << points.size() << std::endl;
    refinedContour.insert(refinedContour.begin(), points.begin(), points.end());
}

void BlobProcessor::filterContours(std::vector<std::vector<cv::Point> > contours, std::vector<std::vector<cv::Point> > filtered)
{
    std::vector<std::vector<cv::Point> > buf;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours.at(i)) > 100)
            buf.push_back(contours.at(i));
    }
    filtered.clear();
    filtered.insert(filtered.begin(), buf.begin(), buf.end());
}

std::vector<uchar> BlobProcessor::sampleLine(cv::Mat mat, cv::Point p1, cv::Point p2) {
    cv::LineIterator iterator = cv::LineIterator(mat, p1, p2);
    std::vector<uchar> sample;
    for (int i = 0; i < iterator.count; ++i) {
        sample.push_back(*iterator.ptr);
        iterator++;
    }
    return sample;
}


/*
 * Copyright by  Nikolas Markou
 * original C# version from:
 * http://nmarkou.blogspot.ru/2012/03/contour-refinement.html
 */

cv::Point2f BlobProcessor::normalAtPoint(cv::Point prev, cv::Point current, cv::Point next, bool inOut) {
    cv::Point2f normal;
    float dx1 = current.x - prev.x,
          dx2 = next.x - current.x,
          dy1 = current.y - prev.y,
          dy2 = next.y - current.y;
    if (inOut)
        normal = cv::Point((dy1 + dy2) * 0.5f, -(dx1 + dx2) * 0.5f);
    else
        normal = cv::Point(-(dy1 + dy2) * 0.5f, (dx1 + dx2) * 0.5f);
    return normalizePoint(normal);
}

/*
 * Copyright by  Nikolas Markou
 * original C# version from:
 * http://nmarkou.blogspot.ru/2012/03/contour-refinement.html
 */

cv::Point2f BlobProcessor::normalizePoint(cv::Point2f point) {
    float length = static_cast<float>(sqrt(point.x * point.x + point.y * point.y));
    if (length > 0.0f)
        return cv::Point2f(point.x / length, point.y / length);
    return cv::Point2f(0.0f, 0.0f);
}

void BlobProcessor::resizeRegions(std::vector<cv::Rect> regions, cv::Size frameSize, size_t delta_x, size_t delta_y,std::vector<cv::Rect>& resized) {
    resized.clear();
    for (size_t r = 0; r < regions.size(); ++r) {
        if (regions.at(r).area() > 100) {
            regions.at(r).x -= delta_x / 2;
            regions.at(r).width += delta_x;
            regions.at(r).y -= delta_y / 2;
            regions.at(r).height += delta_y;
            if (regions.at(r).x < 0)
                regions.at(r).x = 0;
            if (regions.at(r).width + regions.at(r).x >= frameSize.width)
                regions.at(r).width = frameSize.width - regions.at(r).x;
            if (regions.at(r).y < 0)
                regions.at(r).y = 0;
            if (regions.at(r).height + regions.at(r).y >= frameSize.height)
                regions.at(r).height = frameSize.height - regions.at(r).y;
            resized.push_back(regions.at(r));
        }
    }
}

std::vector<cv::Point> BlobProcessor::getMaxContour(cv::Mat map, cv::Point offset) {
    std::vector<std::vector<cv::Point> > contours;
    int maxI = -1, maxArea = 0, currentArea = 0, maxAllowed = map.cols * map.rows * 0.9;
    getContours(map, contours, CV_RETR_EXTERNAL, offset);
    for (size_t c = 0; c < contours.size(); ++c) {
        currentArea = cv::contourArea(contours.at(c));
        if (currentArea > maxArea && currentArea < maxAllowed) {
            maxArea = cv::contourArea(contours.at(c));
            maxI = c;
        }
    }
    if (contours.size() == 1 || (contours.size() > 0 && maxI == -1)) {
        return contours.at(0);
    }
    else {
        return contours.at(maxI);
    }
}

std::vector<cv::Point> BlobProcessor::getMaxContourFromRoi(cv::Mat map, cv::Rect roi) {
    return getMaxContour(map(roi), roi.tl());
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
    return sqrt(pow(static_cast<float>(a.x - b.x), 2.0f) + pow(static_cast<float>(a.y - b.y), 2.0f));
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

cv::Point BlobProcessor::getCenterOfMasses(cv::Mat blobMask) {
    cv::Moments moments = cv::moments(blobMask);
    return cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

cv::Point BlobProcessor::getCenterOfMasses(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    return cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

bool checkPoint(cv::Point p1, cv::Point p2, cv::Mat mat) {
    if (abs(mat.at<uchar>(p1) - mat.at<uchar>(p2)) <= 10)
        return true;
    else
        return false;
}

void BlobProcessor::checkTopBottom(bool &bottom, bool &top, cv::Mat hsv, cv::Point point, std::vector<cv::Point> &points, int x)
{
    if (!top && point.y - 1 > 0 && checkPoint(cv::Point(x, point.y - 1), point, hsv)) {
        top = true;
        points.push_back(cv::Point(x, point.y - 1));
    }
    if (top && point.y - 1 > 0 && !checkPoint(cv::Point(x, point.y - 1), point, hsv)) {
        top = false;
    }
    if (!bottom && point.y + 1 < hsv.rows && checkPoint(cv::Point(x, point.y + 1), point, hsv)) {
        bottom = true;
        points.push_back(cv::Point(x, point.y + 1));
    }
    if (bottom && point.y + 1 < hsv.rows && !checkPoint(cv::Point(x, point.y + 1), point, hsv)) {
        bottom = false;
    }
}

double BlobProcessor::calculateEccentricity(std::vector<cv::Point> contour) {
    cv::Moments moments = cv::moments(contour);
    double a20 = moments.mu20 / moments.m00;
    double a02 = moments.mu02 / moments.m00;
    double a11 = moments.mu11 / moments.m00;
    double l1 = (a20 + a02) / 2 + sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    double l2 = (a20 + a02) / 2 - sqrt(4 * a11 * a11 + (a20 - a02) * (a20 - a02)) / 2;
    return 1 - l2 / l1;

}

void BlobProcessor::growRegions(bool useGray, cv::Mat input, cv::Point start, std::vector<cv::Point> &contour) {
/*    if (useGray) {
        cv::Mat gray;
        cv::cvtColor(input, gray, CV_BGR2GRAY);
        cv::floodFill(gray, start, cv::Scalar(255), 0, cv::Scalar(30), cv::Scalar(30), cv::FLOODFILL_FIXED_RANGE);
        cv::threshold(gray, gray, 254, 255, CV_THRESH_BINARY);
        //cv::imshow("gray", gray);
        contour = getMaxContour(gray(roi), roi.tl());
        //contour = getMaxContour(gray, roi.tl());
    } else {
        cv::Mat ycrcb;
        cv::cvtColor(input, ycrcb, CV_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::floodFill(channels.[1], start, cv::Scalar(255), 0, cv::Scalar(30), cv::Scalar(30), cv::FLOODFILL_FIXED_RANGE);

    }
*/
}

void BlobProcessor::growRegions(bool useGray, cv::Mat input, cv::Rect roi, std::vector<cv::Point> &contour) {
    growRegions(useGray, input, cv::Point(roi.width / 2, roi.height * 0.8), contour);
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


