#include "common.h"
#include "face_detector.h"
#include "model_io.h"
#include "integral_image.h"
#include "vjlearner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#endif

namespace fs = std::filesystem;

// -------------------------------------------------------------
// UI Colors (Modern Dark Theme)
// -------------------------------------------------------------
const cv::Scalar BG_COLOR(28, 26, 24);
const cv::Scalar PANEL_BG(40, 36, 34);
const cv::Scalar CARD_BG(52, 48, 44);
const cv::Scalar BTN_NORMAL(68, 62, 58);
const cv::Scalar BTN_HOVER(90, 82, 75);
const cv::Scalar BTN_ACTIVE(180, 110, 45); // Blue-ish
const cv::Scalar TEXT_WHITE(245, 245, 245);
const cv::Scalar TEXT_DIM(160, 160, 160);
const cv::Scalar ACCENT_CUSTOM(80, 220, 100);  // Green (Custom VJ)
const cv::Scalar ACCENT_CASCADE(0, 215, 255);  // Gold (OpenCV)
const cv::Scalar BORDER_COLOR(65, 60, 56);

// -------------------------------------------------------------
// Camera Helper for Windows Media Foundation
// -------------------------------------------------------------
#ifdef _WIN32
template <typename T> void releaseMF(T *&obj) {
  if (obj) { obj->Release(); obj = nullptr; }
}

class CameraMF {
public:
  ~CameraMF() { close(); }
  bool open() {
    if (m_opened) return true;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) return false;
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) { CoUninitialize(); return false; }
    m_comInit = true;

    IMFAttributes *attrs = nullptr;
    IMFActivate **devs = nullptr;
    UINT32 count = 0;
    IMFMediaSource *src = nullptr;
    IMFAttributes *readerAttrs = nullptr;
    IMFMediaType *mediaType = nullptr;
    IMFMediaType *curMediaType = nullptr;

    bool ok = false;
    do {
      if (FAILED(MFCreateAttributes(&attrs, 1))) break;
      if (FAILED(attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))) break;
      if (FAILED(MFEnumDeviceSources(attrs, &devs, &count)) || count == 0) break;
      if (FAILED(devs[0]->ActivateObject(IID_PPV_ARGS(&src)))) break;
      if (FAILED(MFCreateAttributes(&readerAttrs, 1))) break;
      if (FAILED(readerAttrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE))) break;
      if (FAILED(MFCreateSourceReaderFromMediaSource(src, readerAttrs, &m_reader))) break;
      if (FAILED(MFCreateMediaType(&mediaType))) break;
      if (FAILED(mediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video))) break;
      if (FAILED(mediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32))) break;
      if (FAILED(m_reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, mediaType))) break;
      if (FAILED(m_reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &curMediaType))) break;
      if (FAILED(MFGetAttributeSize(curMediaType, MF_MT_FRAME_SIZE, &m_width, &m_height))) break;
      ok = true;
    } while (false);

    releaseMF(curMediaType);
    releaseMF(mediaType);
    releaseMF(readerAttrs);
    releaseMF(src);
    if (devs) {
      for (UINT32 i = 0; i < count; ++i) releaseMF(devs[i]);
      CoTaskMemFree(devs);
    }
    releaseMF(attrs);
    m_opened = ok;
    return m_opened;
  }

  void close() {
    releaseMF(m_reader);
    if (m_comInit) {
      MFShutdown();
      CoUninitialize();
      m_comInit = false;
    }
    m_opened = false;
  }

  bool isOpened() const { return m_opened; }

  bool read(cv::Mat &frame) {
    if (!m_reader) return false;
    for (int retry = 0; retry < 5; ++retry) {
      DWORD streamIndex = 0, flags = 0;
      LONGLONG timestamp = 0;
      IMFSample *sample = nullptr;
      HRESULT hr = m_reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &timestamp, &sample);
      if (FAILED(hr) || (flags & MF_SOURCE_READERF_ENDOFSTREAM)) return false;
      if (!sample) continue;

      IMFMediaBuffer *buffer = nullptr;
      BYTE *data = nullptr;
      DWORD len = 0;
      bool gotFrame = false;
      if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buffer)) &&
          SUCCEEDED(buffer->Lock(&data, nullptr, &len))) {
        cv::Mat rgba(static_cast<int>(m_height), static_cast<int>(m_width), CV_8UC4, data);
        cv::cvtColor(rgba, frame, cv::COLOR_BGRA2BGR);
        buffer->Unlock();
        gotFrame = true;
      }
      releaseMF(buffer);
      releaseMF(sample);
      if (gotFrame) return true;
    }
    return false;
  }

private:
  IMFSourceReader *m_reader = nullptr;
  UINT32 m_width = 0;
  UINT32 m_height = 0;
  bool m_comInit = false;
  bool m_opened = false;
};
#endif

// -------------------------------------------------------------
// Interactive Button Component
// -------------------------------------------------------------
struct Button {
  std::string id;
  std::string label;
  cv::Rect bounds;
  cv::Scalar color;
  bool active = false;

  bool contains(const cv::Point &pt) const {
    return bounds.contains(pt);
  }

  void draw(cv::Mat &canvas, bool hovered = false) const {
    cv::Scalar bg = active ? BTN_ACTIVE : (hovered ? BTN_HOVER : color);
    cv::rectangle(canvas, bounds, bg, cv::FILLED);
    cv::rectangle(canvas, bounds, BORDER_COLOR, 1);

    int baseline = 0;
    const double fontScale = 0.45;
    const int thickness = 1;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    cv::Point textPos(bounds.x + (bounds.width - textSize.width) / 2,
                      bounds.y + (bounds.height + textSize.height) / 2);
    cv::putText(canvas, label, textPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, TEXT_WHITE, thickness, cv::LINE_AA);
  }
};

// -------------------------------------------------------------
// Application State
// -------------------------------------------------------------
enum DetectionMode { MODE_BOTH = 0, MODE_CUSTOM = 1, MODE_CASCADE = 2 };

struct AppState {
  DetectionMode mode = MODE_BOTH;
  std::string currentImagePath = "test1.bmp";
  cv::Mat currentImage;
  bool webcamActive = false;
  bool showHaarFeatures = false;

  double scaleFactor = 1.25;
  int minVotes = 3;
  int slidingStep = 4;

  std::vector<cv::Rect> customFaces;
  std::vector<cv::Rect> cascadeFaces;
  std::vector<cv::Mat> croppedFaces;

  long long customTimeMs = 0;
  long long cascadeTimeMs = 0;
  double webcamFps = 0.0;
  std::string statusMessage = "He thong san sang.";
  std::string benchmarkSummary = "";

  cv::Point mousePos{0, 0};
  std::string clickedButtonId = "";
};

static AppState g_state;
static std::vector<Button> g_buttons;

// -------------------------------------------------------------
// Model Loaders
// -------------------------------------------------------------
bool loadCustomModel(VJLearner &learner) {
  const std::vector<std::string> paths = {
      "learner.txt", "bin/learner.txt", "../learner.txt", "C:/pbl2/learner.txt"
  };
  for (const auto &p : paths) {
    if (fs::exists(p)) {
      if (loadLearnerTxt(learner, p)) return true;
    }
  }
  return false;
}

bool loadCascade(cv::CascadeClassifier &cascade) {
  return cascade.load("haarcascade/haarcascade_frontalface_alt2.xml") ||
         cascade.load("../haarcascade/haarcascade_frontalface_alt2.xml") ||
         cascade.load("C:/pbl2/haarcascade/haarcascade_frontalface_alt2.xml") ||
         cascade.load("haarcascade/haarcascade_frontalface_default.xml") ||
         cascade.load("../haarcascade/haarcascade_frontalface_default.xml") ||
         cascade.load("C:/pbl2/haarcascade/haarcascade_frontalface_default.xml");
}

std::string openFileDialog() {
#ifdef _WIN32
  char filename[MAX_PATH] = "";
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;
  ofn.lpstrFilter = "Image Files (*.bmp;*.jpg;*.png)\0*.bmp;*.jpg;*.jpeg;*.png\0All Files (*.*)\0*.*\0";
  ofn.lpstrFile = filename;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetOpenFileNameA(&ofn)) {
    return std::string(filename);
  }
#endif
  return "";
}

// -------------------------------------------------------------
// Mouse Callback
// -------------------------------------------------------------
void onMouse(int event, int x, int y, int flags, void *userdata) {
  g_state.mousePos = cv::Point(x, y);
  if (event == cv::EVENT_LBUTTONDOWN) {
    for (const auto &btn : g_buttons) {
      if (btn.contains(g_state.mousePos)) {
        g_state.clickedButtonId = btn.id;
        break;
      }
    }
  }
}

// -------------------------------------------------------------
// Draw Haar Feature Overlay on a face
// -------------------------------------------------------------
void drawHaarOverlay(cv::Mat &display, const cv::Rect &faceRect, const VJLearner &learner) {
  const auto &learners = learner.getWeakLearners();
  if (learners.empty()) return;

  const int numFeaturesToShow = std::min(4, static_cast<int>(learners.size()));
  const double scale = static_cast<double>(faceRect.width) / 24.0;

  for (int k = 0; k < numFeaturesToShow; ++k) {
    const auto &w = learners[k].getWavelet();
    for (int r = 0; r < w.getRows(); ++r) {
      for (int c = 0; c < w.getCols(); ++c) {
        int mPos = w.getMPos() + r * w.getMScaleLength();
        int nPos = w.getNPos() + c * w.getNScaleLength();
        int h = w.getMScaleLength();
        int width = w.getNScaleLength();

        cv::Rect subRect(faceRect.x + static_cast<int>(nPos * scale),
                         faceRect.y + static_cast<int>(mPos * scale),
                         std::max(1, static_cast<int>(width * scale)),
                         std::max(1, static_cast<int>(h * scale)));

        subRect &= cv::Rect(0, 0, display.cols, display.rows);
        if (subRect.width > 0 && subRect.height > 0) {
          bool isPositive = ((r + c) % 2 == 0) ^ w.isNegative();
          cv::Scalar rectColor = isPositive ? cv::Scalar(255, 255, 255) : cv::Scalar(40, 40, 40);
          cv::rectangle(display, subRect, rectColor, cv::FILLED);
          cv::rectangle(display, subRect, cv::Scalar(180, 180, 180), 1);
        }
      }
    }
  }
}

// -------------------------------------------------------------
// Perform Detection on an Image
// -------------------------------------------------------------
void processFrame(cv::Mat &frame, const VJLearner &learner, bool hasCustom,
                  cv::CascadeClassifier &cascade, bool hasCascade) {
  g_state.customFaces.clear();
  g_state.cascadeFaces.clear();
  g_state.croppedFaces.clear();

  cv::Mat gray;
  if (frame.channels() == 1) gray = frame;
  else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 1. Custom Viola-Jones
  if ((g_state.mode == MODE_BOTH || g_state.mode == MODE_CUSTOM) && hasCustom) {
    FaceDetector detector(learner, 24);
    auto t0 = std::chrono::high_resolution_clock::now();
    g_state.customFaces = detector.detect(frame, g_state.slidingStep, g_state.minVotes, 6.0, g_state.scaleFactor);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_state.customTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  }

  // 2. OpenCV Haar Cascade
  if ((g_state.mode == MODE_BOTH || g_state.mode == MODE_CASCADE) && hasCascade) {
    cv::Mat eq;
    cv::equalizeHist(gray, eq);
    auto t0 = std::chrono::high_resolution_clock::now();
    const int minFaceSize = std::max(20, std::min(gray.cols, gray.rows) / 28);
    cascade.detectMultiScale(eq, g_state.cascadeFaces, g_state.scaleFactor, g_state.minVotes,
                             cv::CASCADE_SCALE_IMAGE, cv::Size(minFaceSize, minFaceSize));
    auto t1 = std::chrono::high_resolution_clock::now();
    g_state.cascadeTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  }

  // Collect unique cropped faces
  std::vector<cv::Rect> allFaces;
  if (g_state.mode == MODE_CUSTOM) {
    allFaces = g_state.customFaces;
  } else if (g_state.mode == MODE_CASCADE) {
    allFaces = g_state.cascadeFaces;
  } else {
    allFaces = g_state.customFaces;
    for (const auto &cBox : g_state.cascadeFaces) {
      bool isDuplicate = false;
      for (const auto &uBox : allFaces) {
        int x1 = std::max(cBox.x, uBox.x);
        int y1 = std::max(cBox.y, uBox.y);
        int x2 = std::min(cBox.x + cBox.width, uBox.x + uBox.width);
        int y2 = std::min(cBox.y + cBox.height, uBox.y + uBox.height);
        int iw = std::max(0, x2 - x1);
        int ih = std::max(0, y2 - y1);
        double interArea = iw * ih;
        double minArea = std::min(cBox.area(), uBox.area());
        if (minArea > 0 && (interArea / minArea) > 0.45) {
          isDuplicate = true;
          break;
        }
      }
      if (!isDuplicate) {
        allFaces.push_back(cBox);
      }
    }
  }

  for (const auto &box : allFaces) {
    cv::Rect safeBox = box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.width >= 16 && safeBox.height >= 16) {
      cv::Mat facePatch = frame(safeBox).clone();
      cv::resize(facePatch, facePatch, cv::Size(64, 64));
      g_state.croppedFaces.push_back(facePatch);
    }
  }
}

// -------------------------------------------------------------
// Render Dashboard GUI
// -------------------------------------------------------------
void renderDashboard(cv::Mat &canvas, const cv::Mat &viewportMat, const VJLearner &customLearner) {
  canvas.setTo(BG_COLOR);

  // 1. Header Bar
  cv::rectangle(canvas, cv::Rect(0, 0, 1280, 50), PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, cv::Rect(0, 49, 1280, 1), BORDER_COLOR, cv::FILLED);
  cv::putText(canvas, "PBL2: HE THONG NHAN DIEN KHUON MAT (VIOLA-JONES)",
              cv::Point(20, 32), cv::FONT_HERSHEY_SIMPLEX, 0.65, TEXT_WHITE, 2, cv::LINE_AA);
  cv::putText(canvas, "TRUONG DH BACH KHOA - DH DA NANG",
              cv::Point(950, 32), cv::FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv::LINE_AA);

  // 2. Left Sidebar (Control Panel)
  cv::rectangle(canvas, cv::Rect(15, 60, 320, 645), PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, cv::Rect(15, 60, 320, 645), BORDER_COLOR, 1);

  // Update button bounds
  g_buttons.clear();

  int btnY = 80;
  // Category 1: Nguon dau vao
  cv::putText(canvas, "1. NGUON DAU VAO", cv::Point(30, btnY), cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CASCADE, 1, cv::LINE_AA);
  btnY += 15;
  g_buttons.push_back({"btn_open", "Chon Anh (Open File)", cv::Rect(30, btnY, 290, 34), BTN_NORMAL});
  btnY += 42;
  std::string camLabel = g_state.webcamActive ? "Tat Webcam Live" : "Bat Webcam Live";
  g_buttons.push_back({"btn_webcam", camLabel, cv::Rect(30, btnY, 290, 34), g_state.webcamActive ? BTN_ACTIVE : BTN_NORMAL});
  btnY += 45;

  // Category 2: Che do mo hinh
  cv::putText(canvas, "2. CHON MO HINH NHAN DIEN", cv::Point(30, btnY), cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CASCADE, 1, cv::LINE_AA);
  btnY += 15;
  std::string modeStr = "Che do: SO SANH SONG SONG";
  if (g_state.mode == MODE_CUSTOM) modeStr = "Che do: CUSTOM VIOLA-JONES";
  else if (g_state.mode == MODE_CASCADE) modeStr = "Che do: OPENCV HAAR CASCADE";
  g_buttons.push_back({"btn_mode", modeStr, cv::Rect(30, btnY, 290, 34), BTN_NORMAL});
  btnY += 45;

  // Legend
  cv::circle(canvas, cv::Point(40, btnY), 5, ACCENT_CUSTOM, cv::FILLED);
  cv::putText(canvas, "Custom VJ (Xanh la)", cv::Point(55, btnY + 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
  cv::circle(canvas, cv::Point(190, btnY), 5, ACCENT_CASCADE, cv::FILLED);
  cv::putText(canvas, "OpenCV (Vang)", cv::Point(205, btnY + 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
  btnY += 25;

  // Category 3: Tham so thuat toan
  cv::putText(canvas, "3. THAM SO THUAT TOAN", cv::Point(30, btnY), cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CASCADE, 1, cv::LINE_AA);
  btnY += 15;
  cv::putText(canvas, cv::format("Scale Factor: %.2f", g_state.scaleFactor), cv::Point(35, btnY + 18), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
  g_buttons.push_back({"btn_scale_down", "-", cv::Rect(205, btnY, 50, 26), BTN_NORMAL});
  g_buttons.push_back({"btn_scale_up", "+", cv::Rect(265, btnY, 50, 26), BTN_NORMAL});
  btnY += 34;

  cv::putText(canvas, cv::format("Min Votes: %d", g_state.minVotes), cv::Point(35, btnY + 18), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
  g_buttons.push_back({"btn_votes_down", "-", cv::Rect(205, btnY, 50, 26), BTN_NORMAL});
  g_buttons.push_back({"btn_votes_up", "+", cv::Rect(265, btnY, 50, 26), BTN_NORMAL});
  btnY += 34;

  cv::putText(canvas, cv::format("Sliding Step: %d px", g_state.slidingStep), cv::Point(35, btnY + 18), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
  g_buttons.push_back({"btn_step_down", "-", cv::Rect(205, btnY, 50, 26), BTN_NORMAL});
  g_buttons.push_back({"btn_step_up", "+", cv::Rect(265, btnY, 50, 26), BTN_NORMAL});
  btnY += 45;

  // Category 4: Tinh nang nang cao
  cv::putText(canvas, "4. TINH NANG NANG CAO", cv::Point(30, btnY), cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CASCADE, 1, cv::LINE_AA);
  btnY += 15;
  std::string haarLabel = g_state.showHaarFeatures ? "Dac trung Haar: DANG BAT" : "Dac trung Haar: DANG TAT";
  g_buttons.push_back({"btn_haar", haarLabel, cv::Rect(30, btnY, 290, 32), g_state.showHaarFeatures ? BTN_ACTIVE : BTN_NORMAL});
  btnY += 40;
  g_buttons.push_back({"btn_save", "Luu Anh Ket Qua (Save)", cv::Rect(30, btnY, 290, 32), BTN_NORMAL});
  btnY += 40;
  g_buttons.push_back({"btn_eval", "Kiem Thu Dataset (Benchmark)", cv::Rect(30, btnY, 290, 32), BTN_NORMAL});
  btnY += 50;

  // Hotkey hints
  cv::putText(canvas, "Phim tat: O=Mo anh | W=Cam | M=Che do", cv::Point(25, 680), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv::LINE_AA);
  cv::putText(canvas, "          H=Haar | S=Luu | Q=Thoat", cv::Point(25, 695), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv::LINE_AA);

  // Draw all buttons
  for (const auto &btn : g_buttons) {
    btn.draw(canvas, btn.contains(g_state.mousePos));
  }

  // 3. Main Viewport (Image / Camera Canvas)
  const cv::Rect viewportRect(350, 60, 915, 480);
  cv::rectangle(canvas, viewportRect, cv::Scalar(15, 15, 18), cv::FILLED);
  cv::rectangle(canvas, viewportRect, BORDER_COLOR, 1);

  if (!viewportMat.empty()) {
    cv::Mat displayFrame = viewportMat.clone();

    // Draw Haar feature overlay if enabled
    if (g_state.showHaarFeatures && !g_state.customFaces.empty()) {
      drawHaarOverlay(displayFrame, g_state.customFaces.front(), customLearner);
    }

    // Draw Custom VJ boxes
    if (g_state.mode == MODE_BOTH || g_state.mode == MODE_CUSTOM) {
      for (const auto &box : g_state.customFaces) {
        cv::rectangle(displayFrame, box, ACCENT_CUSTOM, 2);
        cv::putText(displayFrame, "Custom VJ", cv::Point(box.x, std::max(18, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CUSTOM, 1, cv::LINE_AA);
      }
    }

    // Draw OpenCV Cascade boxes
    if (g_state.mode == MODE_BOTH || g_state.mode == MODE_CASCADE) {
      const int offset = (g_state.mode == MODE_BOTH) ? 2 : 0;
      for (const auto &box : g_state.cascadeFaces) {
        cv::Rect adjusted(box.x + offset, box.y + offset, box.width, box.height);
        cv::rectangle(displayFrame, adjusted, ACCENT_CASCADE, 2);
        if (g_state.mode == MODE_BOTH) {
          cv::putText(displayFrame, "OpenCV", cv::Point(box.x, box.y + box.height + 14),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_CASCADE, 1, cv::LINE_AA);
        }
      }
    }

    // Fit image to viewport maintaining aspect ratio
    double scale = std::min(static_cast<double>(viewportRect.width) / displayFrame.cols,
                            static_cast<double>(viewportRect.height) / displayFrame.rows);
    int scaledW = static_cast<int>(displayFrame.cols * scale);
    int scaledH = static_cast<int>(displayFrame.rows * scale);
    cv::Mat resized;
    cv::resize(displayFrame, resized, cv::Size(scaledW, scaledH));

    int offsetX = viewportRect.x + (viewportRect.width - scaledW) / 2;
    int offsetY = viewportRect.y + (viewportRect.height - scaledH) / 2;
    resized.copyTo(canvas(cv::Rect(offsetX, offsetY, scaledW, scaledH)));
  }

  // 4. Bottom Panel: Cropped Faces & Performance
  // Section A: Cropped Faces
  const cv::Rect cropPanel(350, 550, 480, 155);
  cv::rectangle(canvas, cropPanel, PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, cropPanel, BORDER_COLOR, 1);
  cv::putText(canvas, "CAC KHUON MAT PHAT HIEN DUOC (CROPPED FACES)", cv::Point(365, 572),
              cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CASCADE, 1, cv::LINE_AA);

  if (g_state.croppedFaces.empty()) {
    cv::putText(canvas, "(Chua tim thay khuon mat nao trong khung hinh)", cv::Point(410, 635),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv::LINE_AA);
  } else {
    const int maxThumbs = 12;
    const int thumbSize = 54;
    for (size_t i = 0; i < std::min(g_state.croppedFaces.size(), static_cast<size_t>(maxThumbs)); ++i) {
      const auto &thumb = g_state.croppedFaces[i];
      const int col = static_cast<int>(i % 6);
      const int row = static_cast<int>(i / 6);
      const int curX = 365 + col * 76;
      const int curY = 582 + row * 60;
      cv::Mat resizedThumb;
      cv::resize(thumb, resizedThumb, cv::Size(thumbSize, thumbSize));
      resizedThumb.copyTo(canvas(cv::Rect(curX, curY, thumbSize, thumbSize)));
      cv::rectangle(canvas, cv::Rect(curX, curY, thumbSize, thumbSize), ACCENT_CUSTOM, 1);
      cv::putText(canvas, cv::format("#%zu", i + 1), cv::Point(curX + 16, curY + thumbSize - 4),
                  cv::FONT_HERSHEY_SIMPLEX, 0.35, TEXT_WHITE, 1, cv::LINE_AA);
    }
    if (g_state.croppedFaces.size() > maxThumbs) {
      cv::putText(canvas, cv::format("+%zu nua", g_state.croppedFaces.size() - maxThumbs),
                  cv::Point(365 + 5 * 76 + 10, 582 + 60), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv::LINE_AA);
    }
  }

  // Section B: Performance Statistics
  const cv::Rect statsPanel(845, 550, 420, 155);
  cv::rectangle(canvas, statsPanel, PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, statsPanel, BORDER_COLOR, 1);
  cv::putText(canvas, "THONG SO HIEN THI & HIEU NANG", cv::Point(860, 572),
              cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CASCADE, 1, cv::LINE_AA);

  int statY = 596;
  cv::putText(canvas, cv::format("Thoi gian Custom VJ : %lld ms", g_state.customTimeMs),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.40, ACCENT_CUSTOM, 1, cv::LINE_AA);
  statY += 20;
  cv::putText(canvas, cv::format("Thoi gian OpenCV    : %lld ms", g_state.cascadeTimeMs),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.40, ACCENT_CASCADE, 1, cv::LINE_AA);
  statY += 20;

  if (g_state.mode == MODE_BOTH) {
    cv::putText(canvas, cv::format("So mat: Custom %zu | OpenCV %zu (Hop nhat %zu)",
                                   g_state.customFaces.size(), g_state.cascadeFaces.size(), g_state.croppedFaces.size()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv::LINE_AA);
  } else if (g_state.mode == MODE_CUSTOM) {
    cv::putText(canvas, cv::format("So mat tim duoc (Custom): %zu mat", g_state.customFaces.size()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.40, ACCENT_CUSTOM, 1, cv::LINE_AA);
  } else {
    cv::putText(canvas, cv::format("So mat tim duoc (OpenCV): %zu mat", g_state.cascadeFaces.size()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.40, ACCENT_CASCADE, 1, cv::LINE_AA);
  }
  statY += 20;

  if (g_state.webcamActive) {
    cv::putText(canvas, cv::format("Toc do Webcam       : %.1f FPS", g_state.webcamFps),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(0, 255, 128), 1, cv::LINE_AA);
  } else if (!g_state.benchmarkSummary.empty()) {
    cv::putText(canvas, cv::format("Benchmark: %s", g_state.benchmarkSummary.c_str()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(100, 220, 255), 1, cv::LINE_AA);
  } else {
    cv::putText(canvas, cv::format("Trang thai: %s", g_state.statusMessage.c_str()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv::LINE_AA);
  }
  statY += 20;
  cv::putText(canvas, cv::format("Thong bao: %s", g_state.statusMessage.c_str()),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv::LINE_AA);
}

// -------------------------------------------------------------
// Main Function
// -------------------------------------------------------------
int main(int argc, char **argv) {
  std::cout << "Starting Viola-Jones GUI Dashboard..." << std::endl;

  VJLearner customLearner;
  const bool hasCustom = loadCustomModel(customLearner);
  if (hasCustom) {
    std::cout << "[Info] Custom Viola-Jones model loaded successfully." << std::endl;
  } else {
    std::cerr << "[Warning] learner.txt not found! Custom detector will be unavailable." << std::endl;
  }

  cv::CascadeClassifier cascade;
  const bool hasCascade = loadCascade(cascade);
  if (hasCascade) {
    std::cout << "[Info] OpenCV Haar Cascade loaded successfully." << std::endl;
  }

  // Load initial image
  if (argc > 1) {
    g_state.currentImagePath = argv[1];
  }
  g_state.currentImage = cv::imread(g_state.currentImagePath, cv::IMREAD_COLOR);
  if (g_state.currentImage.empty()) {
    g_state.currentImagePath = "test1.bmp";
    g_state.currentImage = cv::imread(g_state.currentImagePath, cv::IMREAD_COLOR);
  }

  if (!g_state.currentImage.empty()) {
    processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
  }

#ifdef _WIN32
  CameraMF camera;
#endif

  const std::string winName = "PBL2: Viola-Jones Face Detection Dashboard";
  cv::namedWindow(winName, cv::WINDOW_NORMAL);
  cv::resizeWindow(winName, 1280, 720);
  cv::setMouseCallback(winName, onMouse, nullptr);

  cv::Mat canvas(720, 1280, CV_8UC3);

  auto fpsTimer = std::chrono::high_resolution_clock::now();
  int frameCounter = 0;
  int webcamFrameCount = 0;

  bool running = true;
  while (running) {
    // 1. Process Clicked Button Events
    if (!g_state.clickedButtonId.empty()) {
      const std::string id = g_state.clickedButtonId;
      g_state.clickedButtonId = "";

      if (id == "btn_open") {
        std::string selected = openFileDialog();
        if (!selected.empty()) {
          cv::Mat loaded = cv::imread(selected, cv::IMREAD_COLOR);
          if (!loaded.empty()) {
            g_state.currentImagePath = selected;
            g_state.currentImage = loaded;
            g_state.webcamActive = false;
#ifdef _WIN32
            if (camera.isOpened()) camera.close();
#endif
            g_state.statusMessage = "Da load anh moi thanh cong.";
            processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
          }
        }
      } else if (id == "btn_webcam") {
#ifdef _WIN32
        if (!g_state.webcamActive) {
          if (camera.open()) {
            g_state.webcamActive = true;
            g_state.statusMessage = "Webcam dang hoat dong.";
          } else {
            g_state.statusMessage = "Khong the ket noi Webcam!";
          }
        } else {
          camera.close();
          g_state.webcamActive = false;
          g_state.statusMessage = "Da tat Webcam.";
          if (!g_state.currentImage.empty()) {
            processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
          }
        }
#endif
      } else if (id == "btn_mode") {
        if (g_state.mode == MODE_BOTH) g_state.mode = MODE_CUSTOM;
        else if (g_state.mode == MODE_CUSTOM) g_state.mode = MODE_CASCADE;
        else g_state.mode = MODE_BOTH;

        if (!g_state.webcamActive && !g_state.currentImage.empty()) {
          processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
        }
      } else if (id == "btn_haar") {
        g_state.showHaarFeatures = !g_state.showHaarFeatures;
      } else if (id == "btn_save") {
        if (!g_state.currentImage.empty()) {
          cv::Mat annotated = g_state.currentImage.clone();

          // Draw Custom VJ detections
          if (g_state.mode == MODE_BOTH || g_state.mode == MODE_CUSTOM) {
            for (const auto &box : g_state.customFaces) {
              cv::rectangle(annotated, box, ACCENT_CUSTOM, 2, cv::LINE_AA);
              cv::putText(annotated, "Custom VJ", cv::Point(box.x, std::max(18, box.y - 6)),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_CUSTOM, 1, cv::LINE_AA);
            }
          }

          // Draw OpenCV Cascade detections
          if (g_state.mode == MODE_BOTH || g_state.mode == MODE_CASCADE) {
            const int offset = (g_state.mode == MODE_BOTH) ? 2 : 0;
            for (const auto &box : g_state.cascadeFaces) {
              cv::Rect adjusted(box.x + offset, box.y + offset, box.width, box.height);
              cv::rectangle(annotated, adjusted, ACCENT_CASCADE, 2, cv::LINE_AA);
              cv::putText(annotated, "OpenCV", cv::Point(box.x, box.y + box.height + 16),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_CASCADE, 1, cv::LINE_AA);
            }
          }

          // Draw Haar Feature overlay if active
          if (g_state.showHaarFeatures && !g_state.customFaces.empty()) {
            drawHaarOverlay(annotated, g_state.customFaces.front(), customLearner);
          }

          // Save timestamped file in output/ directory
          fs::create_directories("output");
          auto nowTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
          std::stringstream ss;
          ss << "output/ketqua_" << std::put_time(std::localtime(&nowTime), "%Y%m%d_%H%M%S") << ".jpg";
          std::string timestampedPath = ss.str();

          cv::imwrite(timestampedPath, annotated);
          cv::imwrite("ket_qua_output.jpg", annotated);
          g_state.statusMessage = "Da luu anh ra: " + timestampedPath;
          std::cout << "[Save] Da luu anh annotated: " << timestampedPath << " & ket_qua_output.jpg" << std::endl;
        } else {
          g_state.statusMessage = "Khong co anh de luu!";
        }
      } else if (id == "btn_scale_up") {
        g_state.scaleFactor = std::min(2.0, g_state.scaleFactor + 0.05);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_scale_down") {
        g_state.scaleFactor = std::max(1.05, g_state.scaleFactor - 0.05);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_votes_up") {
        g_state.minVotes = std::min(15, g_state.minVotes + 1);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_votes_down") {
        g_state.minVotes = std::max(1, g_state.minVotes - 1);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_step_up") {
        g_state.slidingStep = std::min(12, g_state.slidingStep + 1);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_step_down") {
        g_state.slidingStep = std::max(2, g_state.slidingStep - 1);
        if (!g_state.webcamActive) processFrame(g_state.currentImage, customLearner, hasCustom, cascade, hasCascade);
      } else if (id == "btn_eval") {
        g_state.statusMessage = "Dang chay danh gia test dataset...";
        if (hasCustom) {
          int tp = 0, fn = 0, tn = 0, fp = 0;
          if (fs::exists("data/test_positive")) {
            for (const auto &ent : fs::directory_iterator("data/test_positive")) {
              cv::Mat im = cv::imread(ent.path().string(), cv::IMREAD_GRAYSCALE);
              if (im.empty()) continue;
              if (im.cols != 24 || im.rows != 24) cv::resize(im, im, cv::Size(24, 24));
              if (customLearner.predict(IntegralImage(im)) == Prediction::FACE) ++tp;
              else ++fn;
            }
          }
          if (fs::exists("data/test_negative")) {
            for (const auto &ent : fs::directory_iterator("data/test_negative")) {
              cv::Mat im = cv::imread(ent.path().string(), cv::IMREAD_GRAYSCALE);
              if (im.empty()) continue;
              if (im.cols != 24 || im.rows != 24) cv::resize(im, im, cv::Size(24, 24));
              if (customLearner.predict(IntegralImage(im)) == Prediction::NON_FACE) ++tn;
              else ++fp;
            }
          }
          int total = tp + fn + tn + fp;
          if (total > 0) {
            double acc = (100.0 * (tp + tn)) / total;
            double prec = (tp + fp > 0) ? (100.0 * tp / (tp + fp)) : 0.0;
            double rec = (tp + fn > 0) ? (100.0 * tp / (tp + fn)) : 0.0;
            g_state.benchmarkSummary = cv::format("Acc: %.1f%% | Prec: %.1f%% | Rec: %.1f%%", acc, prec, rec);
            g_state.statusMessage = cv::format("Benchmark: %.1f%% (%d/%d dung)", acc, tp + tn, total);
            std::cout << "[Benchmark] TP: " << tp << ", FN: " << fn << ", TN: " << tn << ", FP: " << fp
                      << " -> Accuracy: " << acc << "%, Precision: " << prec << "%, Recall: " << rec << "%" << std::endl;
          }
        }
      }
    }

    // 2. Fetch Frame (Camera or Static Image)
    cv::Mat displayMat;
#ifdef _WIN32
    if (g_state.webcamActive && camera.isOpened()) {
      cv::Mat camFrame;
      if (camera.read(camFrame) && !camFrame.empty()) {
        cv::flip(camFrame, camFrame, 1);
        g_state.currentImage = camFrame.clone();
        frameCounter++;
        webcamFrameCount++;

        auto now = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsTimer).count();
        if (ms >= 1000) {
          g_state.webcamFps = (frameCounter * 1000.0) / ms;
          frameCounter = 0;
          fpsTimer = now;
        }

        // Process every 3 frames for webcam responsiveness
        if (webcamFrameCount % 3 == 0) {
          cv::Mat downscaled;
          cv::resize(camFrame, downscaled, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
          processFrame(downscaled, customLearner, hasCustom, cascade, hasCascade);
          // Scale face coordinates back up to full resolution
          for (auto &box : g_state.customFaces) {
            box.x *= 2; box.y *= 2; box.width *= 2; box.height *= 2;
          }
          for (auto &box : g_state.cascadeFaces) {
            box.x *= 2; box.y *= 2; box.width *= 2; box.height *= 2;
          }

          // Re-crop faces from high-resolution frame for clean thumbnails
          g_state.croppedFaces.clear();
          std::vector<cv::Rect> unifiedBoxes;
          if (g_state.mode == MODE_BOTH) {
            unifiedBoxes = g_state.customFaces;
            for (const auto &cBox : g_state.cascadeFaces) {
              bool dup = false;
              for (const auto &uBox : unifiedBoxes) {
                if ((uBox & cBox).area() > 0.4 * std::min(uBox.area(), cBox.area())) {
                  dup = true;
                  break;
                }
              }
              if (!dup) unifiedBoxes.push_back(cBox);
            }
          } else if (g_state.mode == MODE_CUSTOM) {
            unifiedBoxes = g_state.customFaces;
          } else {
            unifiedBoxes = g_state.cascadeFaces;
          }

          for (const auto &box : unifiedBoxes) {
            cv::Rect safeBox = box & cv::Rect(0, 0, camFrame.cols, camFrame.rows);
            if (safeBox.width >= 16 && safeBox.height >= 16) {
              g_state.croppedFaces.push_back(camFrame(safeBox).clone());
            }
          }
        }
        displayMat = camFrame;
      }
    }
#endif
    if (displayMat.empty()) {
      displayMat = g_state.currentImage;
    }

    // 3. Render Dashboard
    renderDashboard(canvas, displayMat, customLearner);
    cv::imshow(winName, canvas);

    // 4. Keyboard Shortcuts
    int key = cv::waitKey(g_state.webcamActive ? 1 : 25);
    if (key == 27 || key == 'q' || key == 'Q') {
      running = false;
    } else if (key == 'o' || key == 'O') {
      g_state.clickedButtonId = "btn_open";
    } else if (key == 'w' || key == 'W') {
      g_state.clickedButtonId = "btn_webcam";
    } else if (key == 'm' || key == 'M') {
      g_state.clickedButtonId = "btn_mode";
    } else if (key == 'h' || key == 'H') {
      g_state.clickedButtonId = "btn_haar";
    } else if (key == 's' || key == 'S') {
      g_state.clickedButtonId = "btn_save";
    }
  }

#ifdef _WIN32
  if (camera.isOpened()) camera.close();
#endif

  return EXIT_SUCCESS;
}
