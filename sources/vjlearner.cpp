#include "vjlearner.h"
#include <cmath>
#include <iomanip>
#include <iostream>

Prediction VJLearner::predict(const IntegralImage &img) const {
  float total = 0.0f;
  for (size_t i = 0; i < this->m_wLearners.size(); ++i) {
    total += this->m_alphas[i] *
             static_cast<float>(this->m_wLearners[i].predict(img));
    if (i != 0 && i % CASCADE_RESET == 0 && total <= 0.0f) {
      return Prediction::NON_FACE;
    }
  }
  return (total > 0.0f) ? Prediction::FACE : Prediction::NON_FACE;
}

std::vector<Prediction>
VJLearner::predict(const std::vector<IntegralImage> &imgs) const {
  std::vector<Prediction> predictions;
  predictions.reserve(imgs.size());
  for (const auto &img : imgs) {
    predictions.push_back(this->predict(img));
  }
  return predictions;
}

VJLearner::VJLearner(int imageSize, int boostAmount)
    : m_imageSize(imageSize), m_boostAmount(boostAmount) {
  this->m_createWavelets();
}

float VJLearner::error(const std::vector<IntegralImage> &imgs,
                       const std::vector<Prediction> &targets) const {
  if (imgs.empty()) return 0.0f;
  size_t incorrect = 0;
  for (size_t i = 0; i < imgs.size(); ++i) {
    if (this->predict(imgs[i]) != targets[i]) {
      ++incorrect;
    }
  }
  return static_cast<float>(incorrect) / static_cast<float>(imgs.size());
}

void VJLearner::train(const std::vector<IntegralImage> &imgs,
                      const std::vector<Prediction> &targets) {
  this->m_wLearners.clear();
  this->m_alphas.clear();
  this->m_createWavelets();

  const size_t n = imgs.size();
  if (n == 0) return;

  std::vector<float> imgWeights(n, 1.0f / static_cast<float>(n));

  for (int i = 0; i < this->m_boostAmount; i++) {
    std::cout << "\nIteration: " << i << std::endl;
    WLearner learner;
    learner.train(imgs, imgWeights, targets, this->m_wavelets);
    float err = learner.error(imgs, imgWeights, targets);
    std::cout << "Error: " << err << std::endl;

    if (err >= 0.5f) {
      std::cout << "Weak learner error >= 0.5, stopping boost." << std::endl;
      break;
    }

    this->m_wLearners.push_back(learner);
    if (err <= 1e-6f) {
      err = 1e-4f;
    }
    const float alpha = 0.5f * std::log((1.0f - err) / err);
    std::cout << "Alpha: " << alpha << std::endl;
    this->m_alphas.push_back(alpha);

    for (size_t j = 0; j < n; ++j) {
      const Prediction p = learner.predict(imgs[j]);
      if (p == targets[j]) {
        imgWeights[j] *= std::exp(-alpha);
      } else {
        imgWeights[j] *= std::exp(alpha);
      }
    }
    const float sum = std::accumulate(imgWeights.begin(), imgWeights.end(), 0.0f);
    if (sum > 0.0f) {
      std::transform(imgWeights.begin(), imgWeights.end(), imgWeights.begin(),
                     [sum](float val) { return val / sum; });
    }
  }
}

void VJLearner::m_createWavelets() {
  this->m_wavelets.clear();
  for (int m = 0; m < this->m_imageSize; m += this->WAVELET_STRIDE) {
    for (int n = 0; n < this->m_imageSize; n += this->WAVELET_STRIDE) {
      for (int scale : this->WAVELET_BASES) {
        std::vector<Wavelet> candidates;
        candidates.push_back(
            {H1_MLENGTH, H1_NLENGTH, H1_ROWS, H1_COLS, m, n, scale, false});
        candidates.push_back(candidates.back().transpose());
        candidates.push_back(
            {H2_MLENGTH, H2_NLENGTH, H2_ROWS, H2_COLS, m, n, scale, false});
        candidates.push_back(candidates.back().transpose());
        candidates.push_back(
            {H3_MLENGTH, H3_NLENGTH, H3_ROWS, H3_COLS, m, n, scale, false});
        candidates.push_back(candidates.back().transpose());
        candidates.push_back(
            {H4_MLENGTH, H4_NLENGTH, H4_ROWS, H4_COLS, m, n, scale, false});
        candidates.push_back(candidates.back().transpose());

        for (auto &w : candidates) {
          if (w.onImage(this->m_imageSize)) {
            this->m_wavelets.push_back(w);
            this->m_wavelets.push_back(Wavelet(w, true));
          }
        }
      }
    }
  }
  std::cout << "Wavelets: " << this->m_wavelets.size() << std::endl;
}

void to_json(json &j, const VJLearner &l) {
  j = json{{"m_wLearners", l.m_wLearners},
           {"m_alphas", l.m_alphas},
           {"m_boostAmount", l.m_boostAmount},
           {"m_imageSize", l.m_imageSize}};
}

void from_json(const json &j, std::vector<WLearner> &wLearners) {
  for (const auto &item : j) {
    WLearner learner;
    item.get_to(learner);
    wLearners.push_back(learner);
  }
}

void from_json(const json &j, VJLearner &l) {
  j.at("m_alphas").get_to(l.m_alphas);
  j.at("m_wLearners").get_to(l.m_wLearners);
  j.at("m_boostAmount").get_to(l.m_boostAmount);
  j.at("m_imageSize").get_to(l.m_imageSize);
  l.m_createWavelets();
}
