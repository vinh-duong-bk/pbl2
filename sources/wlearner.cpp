#include "wlearner.h"
#include <algorithm>
#include <limits>

void WLearner::train(const std::vector<IntegralImage> &imgs,
                     const std::vector<float> &imgWeights,
                     const std::vector<Prediction> &targets,
                     std::vector<Wavelet> &wavelets) {
  Pool<WErrorInfo> pool;
  for (const auto &w : wavelets) {
    pool.queue(
        [&, w]() { return this->m_bestSplit(w, imgs, imgWeights, targets); });
  }
  std::vector<WErrorInfo> infos = pool.takeReturnVals();

  if (infos.empty()) {
    return;
  }

  const auto bestInfo = *std::min_element(
      infos.begin(), infos.end(),
      [](const WErrorInfo &i1, const WErrorInfo &i2) {
        return i1.error < i2.error;
      });

  this->m_wavelet = bestInfo.w;
  this->m_splitVal = bestInfo.split;
  this->m_polarity = bestInfo.polarity;
}

WLearner::WErrorInfo
WLearner::m_bestSplit(const Wavelet &w, const std::vector<IntegralImage> &imgs,
                      const std::vector<float> &imgWeights,
                      const std::vector<Prediction> &targets) {
  const size_t n = imgs.size();
  std::vector<WSplitInfo> splitInfos;
  splitInfos.reserve(n);

  float totalFaceWeight = 0.0f;
  float totalNonFaceWeight = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    const auto val = w.getWaveVal(imgs[i]);
    const float weight = imgWeights[i];
    const Prediction target = targets[i];
    splitInfos.push_back({weight, target, val});
    if (target == Prediction::FACE) {
      totalFaceWeight += weight;
    } else {
      totalNonFaceWeight += weight;
    }
  }

  std::sort(splitInfos.begin(), splitInfos.end(),
            [](const WSplitInfo &w1, const WSplitInfo &w2) {
              return w1.waveVal < w2.waveVal;
            });

  float runningFaceWeight = 0.0f;
  float runningNonFaceWeight = 0.0f;

  float minError = std::numeric_limits<float>::infinity();
  Wavelet::waveVal bestSplit = 0;
  int bestPolarity = 1;

  for (size_t i = 0; i < n; ++i) {
    if (splitInfos[i].prediction == Prediction::FACE) {
      runningFaceWeight += splitInfos[i].imgWeight;
    } else {
      runningNonFaceWeight += splitInfos[i].imgWeight;
    }

    if (i + 1 < n && splitInfos[i].waveVal == splitInfos[i + 1].waveVal) {
      continue;
    }

    // Polarity +1: f(x) > theta -> FACE, f(x) <= theta -> NON_FACE
    const float errPos = runningFaceWeight + (totalNonFaceWeight - runningNonFaceWeight);

    // Polarity -1: f(x) <= theta -> FACE (i.e. -1 * f(x) >= -1 * theta)
    const float errNeg = runningNonFaceWeight + (totalFaceWeight - runningFaceWeight);

    if (errPos < minError) {
      minError = errPos;
      bestSplit = splitInfos[i].waveVal;
      bestPolarity = 1;
    }
    if (errNeg < minError) {
      minError = errNeg;
      bestSplit = splitInfos[i].waveVal;
      bestPolarity = -1;
    }
  }

  return {w, minError, bestSplit, bestPolarity};
}

Prediction WLearner::predict(const IntegralImage &img) const {
  const auto val = m_wavelet.getWaveVal(img);
  if (m_polarity * val > m_polarity * m_splitVal) {
    return Prediction::FACE;
  }
  return Prediction::NON_FACE;
}

std::vector<Prediction>
WLearner::predict(const std::vector<IntegralImage> &imgs) const {
  std::vector<Prediction> predictions;
  predictions.reserve(imgs.size());
  for (const auto &img : imgs) {
    predictions.push_back(this->predict(img));
  }
  return predictions;
}

float WLearner::error(const std::vector<IntegralImage> &imgs,
                      const std::vector<float> &imgWeights,
                      const std::vector<Prediction> &targets) const {
  float err = 0.0f;
  for (size_t i = 0; i < imgs.size(); ++i) {
    if (this->predict(imgs[i]) != targets[i]) {
      err += imgWeights[i];
    }
  }
  return err;
}

void to_json(json &j, const WLearner &wL) {
  j = json{{"m_splitVal", wL.m_splitVal},
           {"m_wavelet", wL.m_wavelet},
           {"m_polarity", wL.m_polarity}};
}

void from_json(const json &j, WLearner &wL) {
  j.at("m_splitVal").get_to(wL.m_splitVal);
  j.at("m_wavelet").get_to(wL.m_wavelet);
  if (j.contains("m_polarity")) {
    j.at("m_polarity").get_to(wL.m_polarity);
  } else {
    wL.m_polarity = 1;
  }
}
