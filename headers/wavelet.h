#pragma once

#include "common.h"
#include "integral_image.h"

class Wavelet {

public:
  typedef long waveVal;

  Wavelet() = default;

  Wavelet(int mLength, int nLength, int rows, int cols, int mPos, int nPos,
          int scale, bool isNegative);

  Wavelet(const Wavelet &other, bool isNegative);

  Wavelet transpose();

  waveVal getWaveVal(const IntegralImage &img) const;

  bool onImage(int imageLength);

  int getMPos() const { return m_mPos; }
  int getNPos() const { return m_nPos; }
  int getRows() const { return m_rows; }
  int getCols() const { return m_cols; }
  int getMScaleLength() const { return m_mScaleLength; }
  int getNScaleLength() const { return m_nScaleLength; }
  bool isNegative() const { return m_isNegative; }

  friend void to_json(json &j, const Wavelet &w);

  friend void from_json(const json &j, Wavelet &w);

private:
  int m_mScaleLength = 0;

  int m_nScaleLength = 0;

  int m_rows = 0;

  int m_cols = 0;

  int m_nPos = 0;

  int m_mPos = 0;

  bool m_isNegative = false;
};
