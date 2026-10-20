#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
const cv::Scalar PANEL_BG(34, 38, 44);
const cv::Scalar CARD_BG(52, 48, 44);
const cv::Scalar BTN_NORMAL(67, 74, 84);
const cv::Scalar BTN_HOVER(90, 101, 114);
const cv::Scalar BTN_ACTIVE(190, 125, 42);
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
    const double fontScale = 0.46;
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
enum DetectionMode { MODE_CASCADE = 0, MODE_CUSTOM = 1 };

struct AppState {
  DetectionMode mode = MODE_CASCADE;
  std::string currentImagePath;
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
  std::string statusMessage = "SDL2 Graphic UI san sang.";

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

bool loadFaceCascade(cv::CascadeClassifier &cascade) {
  const std::vector<std::string> cascadeFiles = {
      "haarcascade/haarcascade_frontalface_alt2.xml",
      "haarcascade/haarcascade_frontalface_default.xml",
      "../haarcascade/haarcascade_frontalface_alt2.xml",
      "../haarcascade/haarcascade_frontalface_default.xml",
      "C:/pbl2/haarcascade/haarcascade_frontalface_alt2.xml",
      "C:/pbl2/haarcascade/haarcascade_frontalface_default.xml"
  };
  for (const auto &p : cascadeFiles) {
    if (fs::exists(p) && cascade.load(p)) {
      return true;
    }
  }
  return false;
}

void limitImageSize(cv::Mat &image, int maxWidth = 640, int maxHeight = 480) {
  if (image.empty() || (image.cols <= maxWidth && image.rows <= maxHeight)) return;
  const double scale = std::min(static_cast<double>(maxWidth) / image.cols,
                                static_cast<double>(maxHeight) / image.rows);
  cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
}

void updateCroppedFaces(const cv::Mat &frame, const std::vector<cv::Rect> &faces) {
  g_state.croppedFaces.clear();
  if (frame.empty()) return;

  const cv::Rect imageBounds(0, 0, frame.cols, frame.rows);
  for (const cv::Rect &box : faces) {
    const cv::Rect safeBox = box & imageBounds;
    if (safeBox.width < 16 || safeBox.height < 16) continue;

    cv::Mat facePatch = frame(safeBox).clone();
    cv::resize(facePatch, facePatch, cv::Size(64, 64), 0, 0, cv::INTER_AREA);
    g_state.croppedFaces.push_back(std::move(facePatch));
  }
}

std::vector<cv::Rect> reconcileCustomFaces(
    const std::vector<cv::Rect> &customFaces,
    const std::vector<cv::Rect> &cascadeFaces) {
  std::vector<cv::Rect> faces;
  for (const cv::Rect &reference : cascadeFaces) {
    const cv::Point referenceCenter(reference.x + reference.width / 2,
                                    reference.y + reference.height / 2);
    const cv::Rect search(reference.x - reference.width / 2,
                          reference.y - reference.height / 2,
                          reference.width * 2, reference.height * 2);
    const int referenceArea = std::max(1, reference.area());
    const int referenceWidth = reference.width;
    const auto match = std::find_if(
        customFaces.begin(), customFaces.end(),
        [&search, &referenceCenter, referenceArea, referenceWidth](
            const cv::Rect &candidate) {
          const cv::Point candidateCenter(candidate.x + candidate.width / 2,
                                          candidate.y + candidate.height / 2);
          const double centerDistance = cv::norm(candidateCenter - referenceCenter);
          const double sizeRatio =
              static_cast<double>(candidate.area()) / referenceArea;
          return search.contains(candidateCenter) && sizeRatio > 0.25 &&
                 sizeRatio < 4.0 && centerDistance < referenceWidth;
        });
    if (match != customFaces.end()) {
      // Use the cascade geometry for the final box. Custom detections are
      // used as confirmation only because their sliding-window size can
      // cover the whole body when the face is near a larger window.
      faces.push_back(reference);
    }
  }
  return faces;
}

void detectCascadeFaces(const cv::Mat &gray, cv::CascadeClassifier &cascade,
                        std::vector<cv::Rect> &faces, double scaleFactor,
                        int minNeighbors) {
  cv::Mat equalized;
  cv::equalizeHist(gray, equalized);
  const int minimumFaceSize = std::max(16, std::min(gray.cols, gray.rows) / 32);
  cascade.detectMultiScale(equalized, faces, scaleFactor, minNeighbors,
                           cv::CASCADE_SCALE_IMAGE,
                           cv::Size(minimumFaceSize, minimumFaceSize));
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
                  int detectionStep = -1, double maxScale = -1.0,
                  bool preferSingleFace = false,
                  cv::CascadeClassifier *cascade = nullptr) {
  (void)preferSingleFace;
  g_state.customFaces.clear();
  g_state.cascadeFaces.clear();
  g_state.croppedFaces.clear();

  cv::Mat gray;
  if (frame.channels() == 1) gray = frame;
  else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  if (g_state.mode == MODE_CASCADE && cascade) {
    auto t0 = std::chrono::high_resolution_clock::now();
    detectCascadeFaces(gray, *cascade, g_state.cascadeFaces,
                       g_state.scaleFactor, g_state.minVotes);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_state.cascadeTimeMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  }

  if (g_state.mode == MODE_CUSTOM && hasCustom) {
    FaceDetector detector(learner, 24);
    auto t0 = std::chrono::high_resolution_clock::now();
    const int step = g_state.slidingStep;
    const double scale = 4.0;
    g_state.customFaces = detector.detect(frame, step, g_state.minVotes, scale, g_state.scaleFactor);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_state.customTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  }

  const std::vector<cv::Rect> &allFaces =
      g_state.mode == MODE_CASCADE ? g_state.cascadeFaces : g_state.customFaces;
  updateCroppedFaces(frame, allFaces);
}

void drawEmptyState(cv::Mat &canvas, const cv::Rect &area) {
  const cv::Point center(area.x + area.width / 2, area.y + area.height / 2 - 20);
  cv::rectangle(canvas, area, cv::Scalar(22, 24, 28), cv::FILLED);

  const int faceSize = 150;
  const cv::Rect face(center.x - faceSize / 2, center.y - faceSize / 2,
                      faceSize, faceSize);
  cv::rectangle(canvas, face, cv::Scalar(58, 64, 72), 2, cv::LINE_AA);
  cv::circle(canvas, cv::Point(center.x - 35, center.y - 18), 8,
             ACCENT_CUSTOM, cv::FILLED);
  cv::circle(canvas, cv::Point(center.x + 35, center.y - 18), 8,
             ACCENT_CUSTOM, cv::FILLED);
  cv::ellipse(canvas, cv::Point(center.x, center.y + 28), cv::Size(42, 22),
              0, 0, 180, ACCENT_CASCADE, 3, cv::LINE_AA);

  for (int x = face.x - 34; x <= face.x + face.width + 20; x += 42) {
    cv::line(canvas, cv::Point(x, face.y - 32), cv::Point(x, face.y - 8),
             cv::Scalar(82, 90, 100), 1, cv::LINE_AA);
  }
  for (int y = face.y - 32; y <= face.y + face.height + 24; y += 42) {
    cv::line(canvas, cv::Point(face.x - 34, y), cv::Point(face.x - 10, y),
             cv::Scalar(82, 90, 100), 1, cv::LINE_AA);
  }

  cv::putText(canvas, "VIOLA-JONES", cv::Point(center.x - 92, center.y + 125),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, TEXT_WHITE, 2, cv::LINE_AA);
  cv::putText(canvas, "Chon anh hoac bat webcam de bat dau", 
              cv::Point(center.x - 190, center.y + 158),
              cv::FONT_HERSHEY_SIMPLEX, 0.52, TEXT_DIM, 1, cv::LINE_AA);
}

// -------------------------------------------------------------
// Render Dashboard GUI
// -------------------------------------------------------------
void renderDashboard(cv::Mat &canvas, const cv::Mat &viewportMat, const VJLearner &customLearner) {
  canvas.setTo(BG_COLOR);

  // 1. Header Bar
  cv::rectangle(canvas, cv::Rect(0, 0, 1280, 50), PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, cv::Rect(0, 49, 1280, 1), BORDER_COLOR, cv::FILLED);
  cv::putText(canvas, "PBL2: HE THONG NHAN DIEN KHUON MAT (VIOLA-JONES - SDL2 GRAPHICS)",
              cv::Point(20, 32), cv::FONT_HERSHEY_SIMPLEX, 0.65, TEXT_WHITE, 2, cv::LINE_AA);
  cv::putText(canvas, "TRUONG DH BACH KHOA - DH DA NANG",
              cv::Point(950, 32), cv::FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv::LINE_AA);

  // 2. Left Sidebar (Control Panel)
  cv::rectangle(canvas, cv::Rect(15, 60, 320, 645), PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, cv::Rect(15, 60, 320, 645), BORDER_COLOR, 1);

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
  std::string modeStr = g_state.mode == MODE_CUSTOM
      ? "Che do: CUSTOM VIOLA-JONES"
      : "Che do: OPENCV HAAR CASCADE";
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
  cv::putText(canvas, cv::format("OpenCV Scale Factor: %.2f", g_state.scaleFactor), cv::Point(35, btnY + 18), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv::LINE_AA);
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
  cv::putText(canvas, "          H=Haar | S=Luu | Q/Esc=Thoat", cv::Point(25, 695), cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv::LINE_AA);

  // Draw all buttons
  for (const auto &btn : g_buttons) {
    btn.draw(canvas, btn.contains(g_state.mousePos));
  }

  // 3. Main Viewport (Image / Camera Canvas)
  const cv::Rect viewportRect(350, 60, 915, 480);
  cv::rectangle(canvas, viewportRect, cv::Scalar(15, 15, 18), cv::FILLED);
  cv::rectangle(canvas, viewportRect, BORDER_COLOR, 1);

  if (viewportMat.empty()) {
    drawEmptyState(canvas, viewportRect);
  } else {
    cv::Mat displayFrame = viewportMat.clone();

    // Draw Haar feature overlay if enabled
    if (g_state.showHaarFeatures && !g_state.customFaces.empty()) {
      drawHaarOverlay(displayFrame, g_state.customFaces.front(), customLearner);
    }

    // Draw Custom VJ boxes
    if (g_state.mode == MODE_CUSTOM) {
      for (const auto &box : g_state.customFaces) {
        cv::rectangle(displayFrame, box, ACCENT_CUSTOM, 2);
      }
    }

    // Draw OpenCV Cascade boxes
    if (g_state.mode == MODE_CASCADE) {
      for (const auto &box : g_state.cascadeFaces) {
        cv::rectangle(displayFrame, box, ACCENT_CASCADE, 2);
      }
    }

    // Fit image to viewport maintaining aspect ratio
    double scale = std::min(static_cast<double>(viewportRect.width) / displayFrame.cols,
                            static_cast<double>(viewportRect.height) / displayFrame.rows);
    int scaledW = static_cast<int>(displayFrame.cols * scale);
    int scaledH = static_cast<int>(displayFrame.rows * scale);
    cv::Mat resized;
    cv::Mat sharpened;
    cv::GaussianBlur(displayFrame, sharpened, cv::Size(0, 0), 1.1);
    cv::addWeighted(displayFrame, 1.22, sharpened, -0.22, 0.0, sharpened);
    cv::resize(sharpened, resized, cv::Size(scaledW, scaledH), 0, 0,
           cv::INTER_LANCZOS4);

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
    int thumbX = 365;
    const int thumbY = 585;
    const int maxThumbs = 16;
    const int thumbSize = 48;
    for (size_t i = 0; i < std::min(g_state.croppedFaces.size(), static_cast<size_t>(maxThumbs)); ++i) {
      const auto &thumb = g_state.croppedFaces[i];
      const int column = static_cast<int>(i % 8);
      const int row = static_cast<int>(i / 8);
      thumbX = 365 + column * 58;
      const int currentY = thumbY + row * 58;
      cv::Mat smallThumb;
      cv::resize(thumb, smallThumb, cv::Size(thumbSize, thumbSize), 0, 0, cv::INTER_AREA);
      smallThumb.copyTo(canvas(cv::Rect(thumbX, currentY, thumbSize, thumbSize)));
      cv::rectangle(canvas, cv::Rect(thumbX, currentY, thumbSize, thumbSize),
                    g_state.mode == MODE_CASCADE ? ACCENT_CASCADE : ACCENT_CUSTOM, 1);
      cv::putText(canvas, cv::format("#%zu", i + 1), cv::Point(thumbX + 16, currentY + 56),
                  cv::FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv::LINE_AA);
    }
    if (g_state.croppedFaces.size() > maxThumbs) {
      cv::putText(canvas, cv::format("+%zu nua", g_state.croppedFaces.size() - maxThumbs),
                  cv::Point(thumbX + 5, thumbY + 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv::LINE_AA);
    }
  }

  // Section B: Performance Statistics
  const cv::Rect statsPanel(845, 550, 420, 155);
  cv::rectangle(canvas, statsPanel, PANEL_BG, cv::FILLED);
  cv::rectangle(canvas, statsPanel, BORDER_COLOR, 1);
  cv::putText(canvas, "THONG SO HIEN THI & HIEU NANG", cv::Point(860, 572),
              cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CASCADE, 1, cv::LINE_AA);

  int statY = 600;
  cv::putText(canvas, cv::format("Thoi gian Custom VJ : %lld ms", g_state.customTimeMs),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CUSTOM, 1, cv::LINE_AA);
  statY += 22;
  cv::putText(canvas, cv::format("Thoi gian OpenCV    : %lld ms", g_state.cascadeTimeMs),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_CASCADE, 1, cv::LINE_AA);
  statY += 22;
  const size_t detectedFaceCount = g_state.mode == MODE_CASCADE
      ? g_state.cascadeFaces.size() : g_state.customFaces.size();
  cv::putText(canvas, cv::format("So mat tim duoc     : %zu mat", detectedFaceCount),
              cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv::LINE_AA);
  statY += 22;
  if (g_state.webcamActive) {
    cv::putText(canvas, cv::format("Toc do Webcam       : %.1f FPS", g_state.webcamFps),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv::LINE_AA);
  } else {
    cv::putText(canvas, cv::format("Trang thai: %s", g_state.statusMessage.c_str()),
                cv::Point(865, statY), cv::FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv::LINE_AA);
  }
}

// -------------------------------------------------------------
// Main Function (SDL2 Entry Point)
// -------------------------------------------------------------
int main(int argc, char **argv) {
  std::cout << "==========================================================" << std::endl;
  std::cout << " [PBL2] VIOLA-JONES FACE DETECTION (SDL2 GRAPHICS ENGINE)" << std::endl;
  std::cout << " Truong Dai hoc Bach Khoa - Dai hoc Da Nang" << std::endl;
  std::cout << "==========================================================" << std::endl;

  // Initialize SDL2
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_EVENTS) != 0) {
    std::cerr << "[SDL2 Error] Failed to initialize SDL: " << SDL_GetError() << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "[SDL2] Video subsystem initialized successfully." << std::endl;

  // Create SDL Window
  const int WIN_WIDTH = 1280;
  const int WIN_HEIGHT = 720;
  SDL_Window *sdlWindow = SDL_CreateWindow(
      "PBL2: Viola-Jones Face Detection (SDL2 GUI Dashboard)",
      SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      WIN_WIDTH, WIN_HEIGHT,
      SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI
  );

  if (!sdlWindow) {
    std::cerr << "[SDL2 Error] Could not create window: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return EXIT_FAILURE;
  }

  // Create Hardware Accelerated Renderer
  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
  SDL_Renderer *sdlRenderer = SDL_CreateRenderer(
      sdlWindow, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
  );
  if (!sdlRenderer) {
    sdlRenderer = SDL_CreateRenderer(sdlWindow, -1, 0); // fallback software
  }
  if (!sdlRenderer) {
    std::cerr << "[SDL2 Error] Could not create renderer: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(sdlWindow);
    SDL_Quit();
    return EXIT_FAILURE;
  }

  // Streaming Texture for Frame Presentation
  SDL_Texture *screenTexture = SDL_CreateTexture(
      sdlRenderer,
      SDL_PIXELFORMAT_RGB24,
      SDL_TEXTUREACCESS_STREAMING,
      WIN_WIDTH, WIN_HEIGHT
  );
  if (!screenTexture) {
    std::cerr << "[SDL2 Error] Could not create screen texture: " << SDL_GetError() << std::endl;
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(sdlWindow);
    SDL_Quit();
    return EXIT_FAILURE;
  }

  std::cout << "[SDL2] Graphic Window (1280x720) created and hardware renderer ready." << std::endl;

  // Load Models
  VJLearner customLearner;
  const bool hasCustom = loadCustomModel(customLearner);
  if (hasCustom) {
    std::cout << "[Model] Custom Viola-Jones loaded from learner.txt ("
              << customLearner.getWeakLearners().size() << " weak classifiers)." << std::endl;
  } else {
    std::cerr << "[Warning] learner.txt not found! Custom detector disabled." << std::endl;
  }

  cv::CascadeClassifier faceCascade;
  const bool hasCascade = loadFaceCascade(faceCascade);
  std::cout << "[Model] Haar Cascade: " << (hasCascade ? "loaded." : "unavailable.") << std::endl;

  // Initial Image
  if (argc > 1) {
    const std::string imagePath = argv[1];
    cv::Mat loaded = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (!loaded.empty()) {
      g_state.currentImagePath = imagePath;
      g_state.currentImage = loaded;
      limitImageSize(g_state.currentImage, 1600, 1200);
      std::cout << "[Image] Processing input: " << g_state.currentImagePath << std::endl;
      processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false,
                   hasCascade ? &faceCascade : nullptr);
    } else {
      g_state.statusMessage = "Khong doc duoc anh dau vao.";
    }
  } else {
    g_state.statusMessage = "Chua chon anh hoac webcam.";
  }

#ifdef _WIN32
  CameraMF camera;
#endif

  cv::Mat canvas(WIN_HEIGHT, WIN_WIDTH, CV_8UC3);

  auto fpsTimer = std::chrono::high_resolution_clock::now();
  int frameCounter = 0;
  int webcamFrameCount = 0;

  bool running = true;
  SDL_Event event;

  while (running) {
    // 1. Poll SDL Events
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      } else if (event.type == SDL_MOUSEMOTION) {
        g_state.mousePos = cv::Point(event.motion.x, event.motion.y);
      } else if (event.type == SDL_MOUSEBUTTONDOWN) {
        if (event.button.button == SDL_BUTTON_LEFT) {
          cv::Point clickPt(event.button.x, event.button.y);
          for (const auto &btn : g_buttons) {
            if (btn.contains(clickPt)) {
              g_state.clickedButtonId = btn.id;
              break;
            }
          }
        }
      } else if (event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
          case SDLK_ESCAPE:
          case SDLK_q:
            running = false;
            break;
          case SDLK_o:
            g_state.clickedButtonId = "btn_open";
            break;
          case SDLK_w:
            g_state.clickedButtonId = "btn_webcam";
            break;
          case SDLK_m:
            g_state.clickedButtonId = "btn_mode";
            break;
          case SDLK_h:
            g_state.clickedButtonId = "btn_haar";
            break;
          case SDLK_s:
            g_state.clickedButtonId = "btn_save";
            break;
          default:
            break;
        }
      }
    }

    // 2. Process Button Actions
    if (!g_state.clickedButtonId.empty()) {
      const std::string id = g_state.clickedButtonId;
      g_state.clickedButtonId = "";

      if (id == "btn_open") {
        std::cout << "[Action] Mo hop thoai chon file anh..." << std::endl;
        std::string selected = openFileDialog();
        if (!selected.empty()) {
          cv::Mat loaded = cv::imread(selected, cv::IMREAD_COLOR);
          if (!loaded.empty()) {
            limitImageSize(loaded, 1600, 1200);
            g_state.currentImagePath = selected;
            g_state.currentImage = loaded;
            g_state.webcamActive = false;
#ifdef _WIN32
            if (camera.isOpened()) camera.close();
#endif
            g_state.statusMessage = "Da load anh moi: " + fs::path(selected).filename().string();
            processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false,
                         hasCascade ? &faceCascade : nullptr);
            std::cout << "[Detect] " << selected << " -> Custom VJ: "
                      << g_state.customFaces.size() << " (" << g_state.customTimeMs << "ms) | OpenCV: "
                      << g_state.cascadeFaces.size() << " (" << g_state.cascadeTimeMs << "ms)" << std::endl;
          }
        }
      } else if (id == "btn_webcam") {
#ifdef _WIN32
        if (!g_state.webcamActive) {
          std::cout << "[Action] Khoi dong Webcam qua Windows Media Foundation..." << std::endl;
          if (camera.open()) {
            g_state.webcamActive = true;
            webcamFrameCount = 0;
            g_state.statusMessage = "Webcam Live dang hoat dong.";
            std::cout << "[Webcam] Stream video ket noi thanh cong." << std::endl;
          } else {
            g_state.statusMessage = "Khong the ket noi Webcam!";
            std::cerr << "[Webcam] Loi ket noi camera!" << std::endl;
          }
        } else {
          camera.close();
          g_state.webcamActive = false;
          g_state.statusMessage = "Da tat Webcam.";
          std::cout << "[Webcam] Da tat stream camera." << std::endl;
          g_state.customFaces.clear();
          g_state.cascadeFaces.clear();
          g_state.croppedFaces.clear();
        }
#endif
      } else if (id == "btn_mode") {
        g_state.mode = g_state.mode == MODE_CUSTOM ? MODE_CASCADE : MODE_CUSTOM;

        const char *mName = g_state.mode == MODE_CUSTOM
            ? "Custom Viola-Jones" : "OpenCV Haar Cascade";
        std::cout << "[Mode] Chuyen sang che do: " << mName << std::endl;

        if (g_state.webcamActive) {
          webcamFrameCount = 0;
        } else if (!g_state.currentImage.empty()) {
          const auto &faces = g_state.mode == MODE_CASCADE
                                  ? g_state.cascadeFaces
                                  : g_state.customFaces;
          updateCroppedFaces(g_state.currentImage, faces);
        }
      } else if (id == "btn_haar") {
        g_state.showHaarFeatures = !g_state.showHaarFeatures;
        std::cout << "[Haar] Che do truc quan hoa dac trung Haar: "
                  << (g_state.showHaarFeatures ? "BAT" : "TAT") << std::endl;
      } else if (id == "btn_save") {
        std::string outName = "ket_qua_output.jpg";
        if (!g_state.currentImage.empty()) {
          cv::imwrite(outName, g_state.currentImage);
          g_state.statusMessage = "Da luu anh ket qua ra ket_qua_output.jpg";
          std::cout << "[Save] Da luu anh ket qua vao: " << outName << std::endl;
        }
      } else if (id == "btn_scale_up") {
        g_state.scaleFactor = std::min(2.0, g_state.scaleFactor + 0.05);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_scale_down") {
        g_state.scaleFactor = std::max(1.05, g_state.scaleFactor - 0.05);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_votes_up") {
        g_state.minVotes = std::min(15, g_state.minVotes + 1);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_votes_down") {
        g_state.minVotes = std::max(1, g_state.minVotes - 1);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_step_up") {
        g_state.slidingStep = std::min(12, g_state.slidingStep + 1);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_step_down") {
        g_state.slidingStep = std::max(2, g_state.slidingStep - 1);
        if (!g_state.webcamActive && !g_state.currentImage.empty()) processFrame(g_state.currentImage, customLearner, hasCustom, 6, 4.0, false, hasCascade ? &faceCascade : nullptr);
      } else if (id == "btn_eval") {
        g_state.statusMessage = "Dang chay benchmark dataset...";
        std::cout << "[Benchmark] Dang kiem thu tren data/test_positive va data/test_negative..." << std::endl;
        if (hasCustom) {
          int correct = 0, total = 0;
          for (const auto &grp : {std::pair{"data/test_positive", Prediction::FACE},
                                  std::pair{"data/test_negative", Prediction::NON_FACE}}) {
            if (fs::exists(grp.first)) {
              for (const auto &ent : fs::directory_iterator(grp.first)) {
                cv::Mat im = cv::imread(ent.path().string(), cv::IMREAD_GRAYSCALE);
                if (im.empty()) continue;
                cv::resize(im, im, cv::Size(24, 24));
                correct += (customLearner.predict(IntegralImage(im)) == grp.second);
                total++;
              }
            }
          }
          if (total > 0) {
            double acc = (100.0 * correct) / total;
            g_state.statusMessage = cv::format("Do chinh xac: %.1f%% (%d/%d)", acc, correct, total);
            std::cout << "[Benchmark] Hoan tat: Do chinh xac = " << acc << "% (" << correct << "/" << total << " mau)" << std::endl;
          }
        }
      }
    }

    // 3. Fetch Frame (Camera or Static Image)
    cv::Mat displayMat;
#ifdef _WIN32
    if (g_state.webcamActive && camera.isOpened()) {
      cv::Mat camFrame;
      if (camera.read(camFrame) && !camFrame.empty()) {
        cv::flip(camFrame, camFrame, 1);
        limitImageSize(camFrame, 640, 480);
        frameCounter++;
        ++webcamFrameCount;
        auto now = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsTimer).count();
        if (ms >= 1000) {
          g_state.webcamFps = (frameCounter * 1000.0) / ms;
          frameCounter = 0;
          fpsTimer = now;
        }

        if (webcamFrameCount % 6 == 1) {
          cv::Mat detectionFrame;
          cv::resize(camFrame, detectionFrame, cv::Size(), 0.5, 0.5,
                     cv::INTER_AREA);
          processFrame(detectionFrame, customLearner, hasCustom, 3, 4.0, false,
                       hasCascade ? &faceCascade : nullptr);
          for (auto &face : g_state.customFaces) {
            face.x *= 2;
            face.y *= 2;
            face.width *= 2;
            face.height *= 2;
          }
          for (auto &face : g_state.cascadeFaces) {
            face.x *= 2;
            face.y *= 2;
            face.width *= 2;
            face.height *= 2;
          }
        }
        const std::vector<cv::Rect> &webcamFaces =
            g_state.mode == MODE_CASCADE ? g_state.cascadeFaces : g_state.customFaces;
        updateCroppedFaces(camFrame, webcamFaces);
        displayMat = camFrame;
      }
    }
#endif
    if (displayMat.empty()) {
      displayMat = g_state.currentImage;
    }

    // 4. Render Dashboard to Canvas
    renderDashboard(canvas, displayMat, customLearner);

    // 5. Update SDL2 Texture and Present
    cv::Mat rgbCanvas;
    cv::cvtColor(canvas, rgbCanvas, cv::COLOR_BGR2RGB);
    if (SDL_UpdateTexture(screenTexture, nullptr, rgbCanvas.data,
                          static_cast<int>(rgbCanvas.step)) != 0) {
      std::cerr << "[SDL2 Error] Could not update screen texture: " << SDL_GetError() << std::endl;
      running = false;
    }
    SDL_RenderClear(sdlRenderer);
    if (SDL_RenderCopy(sdlRenderer, screenTexture, nullptr, nullptr) != 0) {
      std::cerr << "[SDL2 Error] Could not copy screen texture: " << SDL_GetError() << std::endl;
      running = false;
    }
    SDL_RenderPresent(sdlRenderer);

    // Short sleep if not running webcam to prevent 100% CPU spinning
    if (!g_state.webcamActive) {
      SDL_Delay(16); // ~60 FPS
    }
  }

#ifdef _WIN32
  if (camera.isOpened()) camera.close();
#endif

  // Clean up SDL2
  SDL_DestroyTexture(screenTexture);
  SDL_DestroyRenderer(sdlRenderer);
  SDL_DestroyWindow(sdlWindow);
  SDL_Quit();

  std::cout << "[SDL2] Da dong ung dung va giai phong tai nguyen thanh cong." << std::endl;
  return EXIT_SUCCESS;
}
