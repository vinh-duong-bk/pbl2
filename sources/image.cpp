#include "common.h"
#include "face_detector.h"
#include "model_io.h"
#include "vjlearner.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace fs = std::filesystem;

namespace {

std::string chooseImageFile() {
#ifdef _WIN32
  char filename[MAX_PATH] = "";
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;
  ofn.lpstrFilter = "Image Files (*.bmp;*.jpg;*.jpeg;*.png)\0*.bmp;*.jpg;*.jpeg;*.png\0All Files (*.*)\0*.*\0";
  ofn.lpstrFile = filename;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetOpenFileNameA(&ofn)) {
    return std::string(filename);
  }
#endif
  return "";
}

bool loadFaceCascade(cv::CascadeClassifier &cascade) {
  const std::vector<std::string> cascadeFiles = {
      "haarcascade/haarcascade_frontalface_alt2.xml",
      "haarcascade/haarcascade_frontalface_default.xml",
      "../haarcascade/haarcascade_frontalface_alt2.xml",
      "../haarcascade/haarcascade_frontalface_default.xml",
      "C:/pbl2/haarcascade/haarcascade_frontalface_alt2.xml",
      "C:/pbl2/haarcascade/haarcascade_frontalface_default.xml"
  };
  for (const auto &p : cascadeFiles) {
    if (fs::exists(p) && cascade.load(p)) {
      return true;
    }
  }
  return false;
}

bool loadCustomLearner(VJLearner &learner) {
  const std::vector<std::string> paths = {
      "learner.txt", "bin/learner.txt", "../learner.txt", "C:/pbl2/learner.txt"
  };
  for (const auto &p : paths) {
    if (fs::exists(p)) {
      if (loadLearnerTxt(learner, p)) {
        std::cout << "[Model] Successfully loaded custom model from: " << p << std::endl;
        return true;
      }
      std::cerr << "[Warning] Error reading " << p << std::endl;
    }
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  std::string imagePath;
  std::string mode = "auto"; // "auto", "custom", "cascade", "both"
  std::string outputPath = "ket_qua_output.jpg";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--custom") {
      mode = "custom";
    } else if (arg == "--cascade") {
      mode = "cascade";
    } else if (arg == "--both") {
      mode = "both";
    } else if (arg == "--output" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (arg[0] != '-') {
      imagePath = arg;
    }
  }

  if (imagePath.empty()) {
    std::cout << "No image path provided. Opening file picker..." << std::endl;
    imagePath = chooseImageFile();
    if (imagePath.empty()) {
      imagePath = "test1.bmp";
      std::cout << "Using default sample image: " << imagePath << std::endl;
    } else {
      std::cout << "Selected image: " << imagePath << std::endl;
    }
  }

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Cannot read image from: " << imagePath << std::endl;
    return EXIT_FAILURE;
  }

  // Auto-resize very large images to prevent excessive processing time
  if (image.cols > 1600 || image.rows > 1200) {
    const double ratio = std::min(1600.0 / image.cols, 1200.0 / image.rows);
    cv::resize(image, image, cv::Size(), ratio, ratio, cv::INTER_AREA);
    std::cout << "[Info] Resized image to " << image.cols << "x" << image.rows << " for detection." << std::endl;
  }

  cv::Mat displayImage = image.clone();

  VJLearner customLearner;
  const bool customAvailable = loadCustomLearner(customLearner);

  cv::CascadeClassifier faceCascade;
  const bool cascadeAvailable = loadFaceCascade(faceCascade);

  if (mode == "auto") {
    mode = customAvailable ? "both" : "cascade";
  }

  std::cout << "\n=== Viola-Jones Face Detection ===" << std::endl;
  std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

  // 1. Run Custom Viola-Jones detector
  if ((mode == "custom" || mode == "both") && customAvailable) {
    FaceDetector detector(std::move(customLearner), 24);
    std::cout << "[Running] Custom Viola-Jones Detector..." << std::endl;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto faces = detector.detect(image, 4, 3, 6.0);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "-> Custom VJ detected: " << faces.size()
              << " face(s) in " << durationMs << " ms" << std::endl;

    for (const auto &face : faces) {
      cv::rectangle(displayImage, face, cv::Scalar(0, 255, 0), 2);
      cv::putText(displayImage, "Custom VJ", cv::Point(face.x, std::max(20, face.y - 6)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
  } else if (mode == "custom" && !customAvailable) {
    std::cerr << "[Error] Custom model learner.txt not found! Falling back to cascade." << std::endl;
  }

  // 2. Run OpenCV Haar Cascade
  if (mode == "cascade" || mode == "both") {
    if (cascadeAvailable) {
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      cv::equalizeHist(gray, gray);

      std::vector<cv::Rect> cascadeFaces;
      const auto start = std::chrono::high_resolution_clock::now();
      const int minimumFaceSize =
          std::max(20, std::min(gray.cols, gray.rows) / 28);
      faceCascade.detectMultiScale(
          gray, cascadeFaces, 1.05, 5, cv::CASCADE_SCALE_IMAGE,
          cv::Size(minimumFaceSize, minimumFaceSize));
      const auto end = std::chrono::high_resolution_clock::now();
      const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

      std::cout << "-> OpenCV Haar Cascade detected: " << cascadeFaces.size()
                << " face(s) in " << durationMs << " ms" << std::endl;

      const int offset = (mode == "both") ? 2 : 0;
      for (const auto &face : cascadeFaces) {
        cv::Rect adjustedFace(face.x + offset, face.y + offset, face.width, face.height);
        cv::rectangle(displayImage, adjustedFace, cv::Scalar(255, 200, 0), 2);
        if (mode == "both") {
          cv::putText(displayImage, "OpenCV", cv::Point(face.x, face.y + face.height + 15),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
        }
      }
    } else {
      std::cerr << "[Warning] Could not load OpenCV Haar Cascade XML." << std::endl;
    }
  }

  cv::imwrite(outputPath, displayImage);
  std::cout << "Saved detection result to: " << outputPath << std::endl;

  cv::namedWindow("Viola-Jones Detection", cv::WINDOW_NORMAL);
  cv::resizeWindow("Viola-Jones Detection", std::min(1000, displayImage.cols), std::min(700, displayImage.rows));
  cv::imshow("Viola-Jones Detection", displayImage);
  std::cout << "Press any key to close the window..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
