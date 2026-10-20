#pragma once

#include "vjlearner.h"
#include <opencv2/core.hpp>
#include <vector>

class FaceDetector {
public:
  explicit FaceDetector(VJLearner learner, int windowSize = 24);
  std::vector<cv::Rect> detect(const cv::Mat &image, int step = 4,
                               int minVotes = 8, double maxScale = 8.0,
                               double scaleFactor = 1.25) const;

private:
  VJLearner m_learner;
  int m_windowSize;
};
