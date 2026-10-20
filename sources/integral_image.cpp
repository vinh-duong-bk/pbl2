#include "integral_image.h"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

IntegralImage::IntegralImage(const cv::Mat &img)
    : m_rows(img.rows), m_cols(img.cols), m_storageRows(img.rows),
      m_storageCols(img.cols) {
  if (img.empty()) {
    throw std::invalid_argument("Cannot create an integral image from an empty image");
  }

  cv::Mat gray;
  if (img.channels() == 1) {
    gray = img;
  } else {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  }

  m_stride = gray.cols + 1;
  auto data = std::make_shared<std::vector<uint32_t>>((gray.rows + 1) * m_stride, 0);
  auto &table = *data;

  for (int r = 0; r < gray.rows; ++r) {
    const uchar *rowPtr = gray.ptr<uchar>(r);
    uint32_t rowSum = 0;
    const size_t prevRowOffset = static_cast<size_t>(r) * m_stride;
    const size_t currRowOffset = static_cast<size_t>(r + 1) * m_stride;

    for (int c = 0; c < gray.cols; ++c) {
      rowSum += rowPtr[c];
      table[currRowOffset + c + 1] = table[prevRowOffset + c + 1] + rowSum;
    }
  }

  m_integralData = std::move(data);
}

IntegralImage::IntegralImage(const IntegralImage &source, int row, int col,
                             int windowSize, double scale)
    : m_integralData(source.m_integralData), m_stride(source.m_stride),
      m_storageRows(source.m_storageRows), m_storageCols(source.m_storageCols),
      m_originRow(row), m_originCol(col), m_coordinateScale(scale),
      m_rows(windowSize), m_cols(windowSize) {}

uint32_t IntegralImage::getSum(int m, int n, int mLength, int nLength) const {
  if (!m_integralData || mLength <= 0 || nLength <= 0) {
    return 0;
  }

  int r1, r2, c1, c2;
  if (m_coordinateScale == 1.0 && m_originRow == 0 && m_originCol == 0) {
    r1 = std::clamp(m, 0, static_cast<int>(m_storageRows));
    r2 = std::clamp(m + mLength, 0, static_cast<int>(m_storageRows));
    c1 = std::clamp(n, 0, static_cast<int>(m_storageCols));
    c2 = std::clamp(n + nLength, 0, static_cast<int>(m_storageCols));
  } else {
    r1 = m_originRow + static_cast<int>(std::round(m * m_coordinateScale));
    c1 = m_originCol + static_cast<int>(std::round(n * m_coordinateScale));
    r2 = m_originRow + static_cast<int>(std::round((m + mLength) * m_coordinateScale));
    c2 = m_originCol + static_cast<int>(std::round((n + nLength) * m_coordinateScale));

    r1 = std::clamp(r1, 0, static_cast<int>(m_storageRows));
    r2 = std::clamp(r2, 0, static_cast<int>(m_storageRows));
    c1 = std::clamp(c1, 0, static_cast<int>(m_storageCols));
    c2 = std::clamp(c2, 0, static_cast<int>(m_storageCols));
  }

  if (r1 >= r2 || c1 >= c2) {
    return 0;
  }

  const auto &data = *m_integralData;
  const size_t r1Offset = static_cast<size_t>(r1) * m_stride;
  const size_t r2Offset = static_cast<size_t>(r2) * m_stride;

  return data[r2Offset + c2] - data[r1Offset + c2] - data[r2Offset + c1] + data[r1Offset + c1];
}
