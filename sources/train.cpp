#include "common.h"
#include "integral_image.h"
#include "model_database.h"
#include "model_io.h"
#include "vjlearner.h"

namespace fs = std::filesystem;

const int NON_FACE_LIMIT = 800;
const int FACE_LIMIT = 800;
const int ADABOOST_ITERATIONS = 200;
const int IMAGE_SIZE = 24;

int main() {
  const fs::path facePath = fs::current_path() / "data" / "positive";
  const fs::path nonFacePath = fs::current_path() / "data" / "negative";
  if (!fs::exists(facePath) || !fs::exists(nonFacePath)) {
    std::cerr << "Missing training folders: data/positive and data/negative."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<IntegralImage> imgs;
  std::vector<Prediction> targets;

  std::cout << "Loading faces." << std::endl;
  int counter = 0;

  int faceLimit;
  if (FACE_LIMIT == -1) {
    faceLimit = std::distance(fs::directory_iterator(facePath),
                              fs::directory_iterator{});
  } else {
    faceLimit = FACE_LIMIT;
  }

  for (const auto &entry : fs::directory_iterator(facePath)) {
    if (counter++ == faceLimit) {
      break;
    }
    cv::Mat img;
    img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      continue;
    }
    cv::resize(img, img, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
    IntegralImage iimg(img);
    imgs.push_back(iimg);
    targets.push_back(Prediction::FACE);
  }

  std::cout << "Loading non-faces." << std::endl;
  counter = 0;

  int nonFaceLimit;
  if (NON_FACE_LIMIT == -1) {
    nonFaceLimit = std::distance(fs::directory_iterator(nonFacePath),
                   fs::directory_iterator{});
  } else {
    nonFaceLimit = NON_FACE_LIMIT;
  }
  for (const auto &entry : fs::directory_iterator(nonFacePath)) {
    if (counter++ == nonFaceLimit) {
      break;
    }
    cv::Mat img;
    img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      continue;
    }
    cv::resize(img, img, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
    IntegralImage iimg(img);
    imgs.push_back(iimg);
    targets.push_back(Prediction::NON_FACE);
  }

  if (imgs.empty()) {
    std::cerr << "No valid training images found." << std::endl;
    return EXIT_FAILURE;
  }
  VJLearner learner(IMAGE_SIZE, ADABOOST_ITERATIONS);
  learner.train(imgs, targets);
  const std::string modelPath = "learner.txt";
  if (!saveLearnerTxt(learner, modelPath)) {
    std::cerr << "Cannot save model to " << modelPath << std::endl;
    return EXIT_FAILURE;
  }
  const float trainingError = learner.error(imgs, targets);
  const auto &weakLearners = learner.getWeakLearners();
  if (!recordModelMetadata("face_detector.db", modelPath, IMAGE_SIZE,
                           ADABOOST_ITERATIONS,
                           static_cast<int>(weakLearners.size()),
                           trainingError)) {
    std::cerr << "Cannot record model metadata in face_detector.db."
              << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << std::endl
            << "Final error: " << trainingError << std::endl;
  std::cout << std::endl
            << "Final error (cascade): " << learner.error(imgs, targets)
            << std::endl;
}
