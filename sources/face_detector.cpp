#include "face_detector.h"
#include "integral_image.h"
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

double rectIoU(const cv::Rect &a, const cv::Rect &b) {
  int x1 = std::max(a.x, b.x);
  int y1 = std::max(a.y, b.y);
  int x2 = std::min(a.x + a.width, b.x + b.width);
  int y2 = std::min(a.y + a.height, b.y + b.height);
  int w = std::max(0, x2 - x1);
  int h = std::max(0, y2 - y1);
  double interArea = static_cast<double>(w * h);
  double unionArea = static_cast<double>(a.area() + b.area()) - interArea;
  return (unionArea > 0.0) ? (interArea / unionArea) : 0.0;
}

std::vector<cv::Rect> mergeDetections(std::vector<cv::Rect> candidates, int minVotes) {
  if (candidates.empty()) return {};

  if (minVotes <= 1) {
    std::vector<cv::Rect> merged;
    std::vector<bool> visited(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (visited[i]) continue;
      visited[i] = true;
      double sumX = candidates[i].x;
      double sumY = candidates[i].y;
      double sumW = candidates[i].width;
      double sumH = candidates[i].height;
      int count = 1;
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (!visited[j] && rectIoU(candidates[i], candidates[j]) > 0.5) {
          visited[j] = true;
          sumX += candidates[j].x;
          sumY += candidates[j].y;
          sumW += candidates[j].width;
          sumH += candidates[j].height;
          ++count;
        }
      }
      merged.emplace_back(
          static_cast<int>(std::round(sumX / count)),
          static_cast<int>(std::round(sumY / count)),
          static_cast<int>(std::round(sumW / count)),
          static_cast<int>(std::round(sumH / count))
      );
    }
    return merged;
  }

  std::vector<int> labels;
  int numClasses = cv::partition(candidates, labels, [](const cv::Rect &r1, const cv::Rect &r2) {
    return rectIoU(r1, r2) >= 0.25;
  });

  std::vector<cv::Rect> results;
  for (int c = 0; c < numClasses; ++c) {
    double sumX = 0, sumY = 0, sumW = 0, sumH = 0;
    int count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (labels[i] == c) {
        sumX += candidates[i].x;
        sumY += candidates[i].y;
        sumW += candidates[i].width;
        sumH += candidates[i].height;
        count++;
      }
    }
    if (count >= minVotes) {
      results.emplace_back(
          static_cast<int>(std::round(sumX / count)),
          static_cast<int>(std::round(sumY / count)),
          static_cast<int>(std::round(sumW / count)),
          static_cast<int>(std::round(sumH / count))
      );
    }
  }

  return results;
}

} // namespace

FaceDetector::FaceDetector(VJLearner learner, int windowSize)
    : m_learner(std::move(learner)), m_windowSize(windowSize) {}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat &image, int step,
                                           int minVotes, double maxScale,
                                           double scaleFactor) const {
  std::vector<cv::Rect> candidates;
  if (image.empty() || image.cols < m_windowSize || image.rows < m_windowSize) {
    return candidates;
  }

  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  IntegralImage fullIntegral(gray);
  const double actualStepScale = (scaleFactor >= 1.05) ? scaleFactor : 1.25;

  for (double scale = 1.0; scale <= maxScale; scale *= actualStepScale) {
    const int window = static_cast<int>(std::round(m_windowSize * scale));
    if (window > gray.cols || window > gray.rows) break;

    const double actualScale = static_cast<double>(window) / m_windowSize;
    const int currentStep = std::max(2, static_cast<int>(std::round(step * std::max(1.0, scale * 0.4))));

    for (int y = 0; y <= gray.rows - window; y += currentStep) {
      for (int x = 0; x <= gray.cols - window; x += currentStep) {
        IntegralImage subImg(fullIntegral, y, x, m_windowSize, actualScale);
        if (m_learner.predict(subImg) == Prediction::FACE) {
          candidates.emplace_back(x, y, window, window);
        }
      }
    }
  }

  return mergeDetections(candidates, minVotes);
}
