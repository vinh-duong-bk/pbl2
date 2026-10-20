#include "common.h"
#include "integral_image.h"
#include "model_io.h"
#include "vjlearner.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

bool loadModel(VJLearner &learner, std::string &foundPath) {
  const std::vector<std::string> paths = {
      "learner.txt", "bin/learner.txt", "../learner.txt", "C:/pbl2/learner.txt"
  };
  for (const auto &p : paths) {
    if (fs::exists(p)) {
      if (loadLearnerTxt(learner, p)) {
        foundPath = p;
        return true;
      }
    }
  }
  return false;
}

int main() {
  VJLearner learner;
  std::string modelPath;
  if (!loadModel(learner, modelPath)) {
    std::cerr << "[Error] Cannot find or parse learner.txt! Train the model first." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "[Model] Loaded: " << modelPath << std::endl;

  int tp = 0; // True Positive (Face predicted as Face)
  int fn = 0; // False Negative (Face predicted as Non-Face)
  int tn = 0; // True Negative (Non-Face predicted as Non-Face)
  int fp = 0; // False Positive (Non-Face predicted as Face)

  const auto startTime = std::chrono::high_resolution_clock::now();

  const std::vector<std::pair<std::string, Prediction>> testGroups = {
      {"data/test_positive", Prediction::FACE},
      {"data/test_negative", Prediction::NON_FACE}
  };

  for (const auto &[dirPath, expected] : testGroups) {
    if (!fs::exists(dirPath)) {
      std::cerr << "[Warning] Directory not found: " << dirPath << std::endl;
      continue;
    }

    int groupTotal = 0;
    int groupCorrect = 0;

    for (const auto &entry : fs::directory_iterator(dirPath)) {
      if (!entry.is_regular_file()) continue;

      cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
      if (image.empty()) continue;

      if (image.cols != 24 || image.rows != 24) {
        cv::resize(image, image, cv::Size(24, 24));
      }

      IntegralImage iimg(image);
      const Prediction actual = learner.predict(iimg);

      if (expected == Prediction::FACE) {
        if (actual == Prediction::FACE) ++tp;
        else ++fn;
      } else {
        if (actual == Prediction::NON_FACE) ++tn;
        else ++fp;
      }

      if (actual == expected) ++groupCorrect;
      ++groupTotal;
    }

    std::cout << dirPath << ": " << groupCorrect << "/" << groupTotal << " correct ("
              << std::fixed << std::setprecision(1)
              << (groupTotal > 0 ? (100.0 * groupCorrect / groupTotal) : 0.0) << "%)\n";
  }

  const auto endTime = std::chrono::high_resolution_clock::now();
  const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

  const int total = tp + fn + tn + fp;
  if (total == 0) {
    std::cerr << "[Error] No test samples evaluated!" << std::endl;
    return EXIT_FAILURE;
  }

  const double accuracy = 100.0 * (tp + tn) / total;
  const double precision = (tp + fp) > 0 ? (100.0 * tp / (tp + fp)) : 0.0;
  const double recall = (tp + fn) > 0 ? (100.0 * tp / (tp + fn)) : 0.0;
  const double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall / (precision + recall)) : 0.0;

  std::cout << "\n================ EVALUATION REPORT ================\n";
  std::cout << "Total Samples Evaluated: " << total << " in " << elapsedMs << " ms ("
            << (total * 1000.0 / std::max(1LL, static_cast<long long>(elapsedMs))) << " samples/sec)\n";
  std::cout << "Confusion Matrix:\n";
  std::cout << "  TP: " << tp << " | FN: " << fn << " (Actual Faces: " << (tp + fn) << ")\n";
  std::cout << "  FP: " << fp << " | TN: " << tn << " (Actual Non-Faces: " << (fp + tn) << ")\n";
  std::cout << "---------------------------------------------------\n";
  std::cout << "Accuracy : " << std::fixed << std::setprecision(2) << accuracy << " %\n";
  std::cout << "Precision: " << precision << " %\n";
  std::cout << "Recall   : " << recall << " %\n";
  std::cout << "F1-Score : " << f1 << " %\n";
  std::cout << "===================================================\n";

  return EXIT_SUCCESS;
}
