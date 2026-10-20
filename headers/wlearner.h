#pragma once

#include "common.h"
#include "pool.h"
#include "wavelet.h"

class WLearner {

public:
  WLearner() = default;

  void train(const std::vector<IntegralImage> &imgs,
             const std::vector<float> &imgWeights,
             const std::vector<Prediction> &targets,
             std::vector<Wavelet> &wavelets);

  Prediction predict(const IntegralImage &img) const;

  std::vector<Prediction> predict(const std::vector<IntegralImage> &imgs) const;

  float error(const std::vector<IntegralImage> &imgs,
              const std::vector<float> &imgWeights,
              const std::vector<Prediction> &targets) const;

  const Wavelet &getWavelet() const { return m_wavelet; }
  Wavelet::waveVal getSplitVal() const { return m_splitVal; }
  int getPolarity() const { return m_polarity; }

  friend void to_json(json &j, const WLearner &wL);

  friend void from_json(const json &j, WLearner &wL);

private:
  struct WErrorInfo {
    Wavelet w;
    float error = std::numeric_limits<float>::infinity();
    Wavelet::waveVal split = 0;
    int polarity = 1;
  };

  struct WSplitInfo {
    float imgWeight;
    Prediction prediction;
    Wavelet::waveVal waveVal;
  };

  Wavelet m_wavelet;

  Wavelet::waveVal m_splitVal = 0;

  int m_polarity = 1;

  WErrorInfo m_bestSplit(const Wavelet &w,
                         const std::vector<IntegralImage> &imgs,
                         const std::vector<float> &imgWeights,
                         const std::vector<Prediction> &targets);
};
