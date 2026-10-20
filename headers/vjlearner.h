#pragma once

#include "common.h"
#include "integral_image.h"
#include "wavelet.h"
#include "wlearner.h"

class VJLearner {

public:
  VJLearner() = default;

  VJLearner(int imageSize, int boostAmount);

  Prediction predict(const IntegralImage &img) const;

  std::vector<Prediction> predict(const std::vector<IntegralImage> &imgs) const;

  float error(const std::vector<IntegralImage> &imgs,
              const std::vector<Prediction> &targets) const;

  void train(const std::vector<IntegralImage> &imgs,
             const std::vector<Prediction> &targets);

  const std::vector<WLearner> &getWeakLearners() const { return m_wLearners; }
  const std::vector<float> &getAlphas() const { return m_alphas; }
  int getImageSize() const { return m_imageSize; }

  friend void to_json(json &j, const VJLearner &l);

  friend void from_json(const json &j, VJLearner &l);

private:
  const int WAVELET_STRIDE = 2;

  const std::vector<int> WAVELET_BASES = {1, 2, 3, 4, 5};

  const int CASCADE_RESET = 15;

  void m_createWavelets();

  int m_imageSize = 24;

  std::vector<Wavelet> m_wavelets;

  std::vector<WLearner> m_wLearners;

  std::vector<float> m_alphas;

  int m_boostAmount = 0;

  const int H1_MLENGTH = 4, H1_NLENGTH = 1, H1_COLS = 2, H1_ROWS = 1;

  const int H2_MLENGTH = 2, H2_NLENGTH = 2, H2_COLS = 1, H2_ROWS = 2;

  const int H3_MLENGTH = 2, H3_NLENGTH = 1, H3_COLS = 3, H3_ROWS = 1;

  const int H4_MLENGTH = 1, H4_NLENGTH = 2, H4_ROWS = 2, H4_COLS = 2;
};
