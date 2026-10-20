#include "common.h"
#include "face_detector.h"
#include "model_io.h"
#include "vjlearner.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

namespace fs = std::filesystem;

namespace {

bool loadFaceCascade(cv::CascadeClassifier &cascade) {
  return cascade.load("haarcascade/haarcascade_frontalface_default.xml") ||
         cascade.load("../haarcascade/haarcascade_frontalface_default.xml") ||
         cascade.load("C:/pbl2/haarcascade/haarcascade_frontalface_default.xml");
}

bool loadCustomLearner(VJLearner &learner) {
  const std::vector<std::string> paths = {
      "learner.txt", "bin/learner.txt", "../learner.txt", "C:/pbl2/learner.txt"
  };
  for (const auto &p : paths) {
    if (fs::exists(p)) {
      if (loadLearnerTxt(learner, p)) return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  const fs::path imagePath = (argc > 1) ? argv[1] : "test1.bmp";

  cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Cannot read image: " << imagePath.string() << std::endl;
    return EXIT_FAILURE;
  }

  if (image.cols > 800) {
    const double ratio = 800.0 / image.cols;
    cv::resize(image, image, cv::Size(), ratio, ratio);
  }

  cv::Mat outputImage = image.clone();

  // 1. Try Custom Model
  VJLearner customLearner;
  if (loadCustomLearner(customLearner)) {
    FaceDetector detector(std::move(customLearner), 24);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto faces = detector.detect(image, 4, 3, 6.0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "[Custom VJ] Found " << faces.size() << " face(s) in " << ms << " ms." << std::endl;
    for (const auto &face : faces) {
      cv::rectangle(outputImage, face, cv::Scalar(0, 255, 0), 2);
    }
  } else {
    // Fallback to OpenCV Cascade
    cv::CascadeClassifier faceCascade;
    if (loadFaceCascade(faceCascade)) {
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      cv::equalizeHist(gray, gray);

      std::vector<cv::Rect> faces;
      const auto t0 = std::chrono::high_resolution_clock::now();
      const int minimumFaceSize =
          std::max(24, std::min(gray.cols, gray.rows) / 20);
      faceCascade.detectMultiScale(
          gray, faces, 1.1, 6, cv::CASCADE_SCALE_IMAGE,
          cv::Size(minimumFaceSize, minimumFaceSize));
      const auto t1 = std::chrono::high_resolution_clock::now();
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

      std::cout << "[OpenCV Cascade] Found " << faces.size() << " face(s) in " << ms << " ms." << std::endl;
      for (const auto &face : faces) {
        cv::rectangle(outputImage, face, cv::Scalar(255, 200, 0), 2);
      }
    } else {
      std::cerr << "No face detector model or cascade available." << std::endl;
    }
  }

  const fs::path outputDirectory = fs::current_path() / "output";
  std::error_code fileError;
  fs::create_directories(outputDirectory, fileError);

  const fs::path outputPath =
      outputDirectory / ("ketqua_" + imagePath.stem().string() + imagePath.extension().string());

  if (cv::imwrite(outputPath.string(), outputImage)) {
    std::cout << "Saved result to: " << outputPath.string() << std::endl;
  } else {
    cv::imwrite("ket_qua_output.jpg", outputImage);
    std::cout << "Saved result to: ket_qua_output.jpg" << std::endl;
  }

  return EXIT_SUCCESS;
}
