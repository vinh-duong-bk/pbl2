#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <cstdint>

class IntegralImage {
public:
  IntegralImage() = default;

  explicit IntegralImage(const cv::Mat &img);

  IntegralImage(const IntegralImage &source, int row, int col, int windowSize,
                double scale);

  void updateWindow(int row, int col, double scale);

  uint32_t getSum(int m, int n, int mLength, int nLength) const;

  unsigned int getRows() const { return m_rows; }
  unsigned int getCols() const { return m_cols; }
  double getCoordinateScale() const { return m_coordinateScale; }

private:
  std::shared_ptr<const std::vector<uint32_t>> m_integralData;
  unsigned int m_stride = 0;
  unsigned int m_storageRows = 0;
  unsigned int m_storageCols = 0;

  int m_originRow = 0;
  int m_originCol = 0;
  double m_coordinateScale = 1.0;

  unsigned int m_rows = 0;
  unsigned int m_cols = 0;
};
