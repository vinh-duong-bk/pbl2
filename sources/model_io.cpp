#include "model_io.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
constexpr const char *FORMAT_HEADER = "PBL2_MODEL_TXT 1";
}

bool saveLearnerTxt(const VJLearner &learner, const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }

  const json model = learner;
  const auto &alphas = model.at("m_alphas");
  const auto &learners = model.at("m_wLearners");

  out << FORMAT_HEADER << '\n'
      << "IMAGE_SIZE " << model.at("m_imageSize").get<int>() << '\n'
      << "BOOST_AMOUNT " << model.at("m_boostAmount").get<int>() << '\n'
      << "ALPHA_COUNT " << alphas.size() << '\n'
      << std::setprecision(17);
  for (const auto &alpha : alphas) {
    out << "ALPHA " << alpha.get<float>() << '\n';
  }
  out << "LEARNER_COUNT " << learners.size() << '\n';
  for (size_t i = 0; i < learners.size(); ++i) {
    const auto &weak = learners.at(i);
    const auto &wavelet = weak.at("m_wavelet");
    out << "LEARNER " << i << ' '
        << weak.at("m_splitVal").get<Wavelet::waveVal>() << ' '
        << weak.value("m_polarity", 1) << ' '
        << wavelet.at("m_mScaleLength").get<int>() << ' '
        << wavelet.at("m_nScaleLength").get<int>() << ' '
        << wavelet.at("m_rows").get<int>() << ' '
        << wavelet.at("m_cols").get<int>() << ' '
        << wavelet.at("m_nPos").get<int>() << ' '
        << wavelet.at("m_mPos").get<int>() << ' '
        << wavelet.at("m_isNegative").get<bool>() << '\n';
  }
  return static_cast<bool>(out);
}

bool loadLearnerTxt(VJLearner &learner, const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  std::string header;
  if (!std::getline(in, header) || header != FORMAT_HEADER) {
    return false;
  }

  json model;
  json alphas = json::array();
  json weakLearners = json::array();
  std::string key;
  int imageSize = 0;
  int boostAmount = 0;
  size_t alphaCount = 0;
  size_t learnerCount = 0;

  if (!(in >> key >> imageSize) || key != "IMAGE_SIZE" ||
      !(in >> key >> boostAmount) || key != "BOOST_AMOUNT" ||
      !(in >> key >> alphaCount) || key != "ALPHA_COUNT") {
    return false;
  }

  for (size_t i = 0; i < alphaCount; ++i) {
    float alpha = 0.0f;
    if (!(in >> key >> alpha) || key != "ALPHA") {
      return false;
    }
    alphas.push_back(alpha);
  }

  if (!(in >> key >> learnerCount) || key != "LEARNER_COUNT") {
    return false;
  }
  for (size_t i = 0; i < learnerCount; ++i) {
    size_t index = 0;
    Wavelet::waveVal split = 0;
    int polarity = 1;
    int mScaleLength = 0;
    int nScaleLength = 0;
    int rows = 0;
    int cols = 0;
    int nPos = 0;
    int mPos = 0;
    int isNegative = 0;
    if (!(in >> key >> index >> split >> polarity >> mScaleLength >>
          nScaleLength >> rows >> cols >> nPos >> mPos >> isNegative) ||
        key != "LEARNER" || index != i || (isNegative != 0 && isNegative != 1)) {
      return false;
    }
    weakLearners.push_back({
        {"m_splitVal", split},
        {"m_polarity", polarity},
        {"m_wavelet", {
            {"m_mScaleLength", mScaleLength},
            {"m_nScaleLength", nScaleLength},
            {"m_rows", rows},
            {"m_cols", cols},
            {"m_nPos", nPos},
            {"m_mPos", mPos},
            {"m_isNegative", isNegative != 0}
        }}
    });
  }

  if (alphas.size() != weakLearners.size()) {
    return false;
  }
  model = {
      {"m_alphas", alphas},
      {"m_wLearners", weakLearners},
      {"m_boostAmount", boostAmount},
      {"m_imageSize", imageSize}
  };
  try {
    from_json(model, learner);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}
