#include "common.h"
#include "integral_image.h"
#include "vjlearner.h"
#include "face_detector.h"
#include "model_io.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

#ifdef _WIN32
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#endif

namespace fs = std::filesystem;

namespace {
const cv::Scalar FRAME_COLOR(247, 195, 79);

#ifdef _WIN32
template <typename T> void releaseMediaFoundation(T *&object) {
  if (object) {
    object->Release();
    object = nullptr;
  }
}

class MediaFoundationCamera {
public:
  ~MediaFoundationCamera() {
    releaseMediaFoundation(m_reader);
    if (m_initialized) {
      MFShutdown();
      CoUninitialize();
    }
  }

  bool open() {
    HRESULT result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(result)) {
      std::cerr << "COM initialization failed: 0x" << std::hex
                << static_cast<unsigned long>(result) << std::dec << std::endl;
      return false;
    }
    result = MFStartup(MF_VERSION);
    if (FAILED(result)) {
      std::cerr << "Media Foundation startup failed: 0x" << std::hex
                << static_cast<unsigned long>(result) << std::dec << std::endl;
      CoUninitialize();
      return false;
    }
    m_initialized = true;

    IMFAttributes *attributes = nullptr;
    IMFActivate **devices = nullptr;
    IMFMediaSource *source = nullptr;
    IMFAttributes *readerAttributes = nullptr;
    UINT32 deviceCount = 0;
    IMFMediaType *mediaType = nullptr;
    IMFMediaType *currentMediaType = nullptr;
    bool opened = false;
    auto check = [](HRESULT value, const char *step) {
      if (SUCCEEDED(value)) return true;
      std::cerr << step << " failed: 0x" << std::hex
                << static_cast<unsigned long>(value) << std::dec << std::endl;
      return false;
    };
    do {
      if (!check(MFCreateAttributes(&attributes, 1), "MFCreateAttributes") ||
          !check(attributes->SetGUID(
              MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
              MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
              "Set camera attributes") ||
          !check(MFEnumDeviceSources(attributes, &devices, &deviceCount),
                 "MFEnumDeviceSources"))
        break;
      if (deviceCount == 0) {
        std::cerr << "No camera device was found." << std::endl;
        break;
      }
      if (!check(devices[0]->ActivateObject(IID_PPV_ARGS(&source)),
                 "Activate camera") ||
          !check(MFCreateAttributes(&readerAttributes, 1),
                 "Create reader attributes") ||
          !check(readerAttributes->SetUINT32(
                     MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE),
                 "Enable video conversion") ||
          !check(MFCreateSourceReaderFromMediaSource(source, readerAttributes,
                                                     &m_reader),
                 "Create source reader") ||
          !check(MFCreateMediaType(&mediaType), "MFCreateMediaType") ||
          !check(mediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video),
                 "Set video type") ||
          !check(mediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32),
                 "Set RGB32 type") ||
          !check(m_reader->SetCurrentMediaType(
                     MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, mediaType),
                 "Set camera format") ||
            !check(m_reader->GetCurrentMediaType(
                 MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentMediaType),
               "Get camera format") ||
            !check(MFGetAttributeSize(currentMediaType, MF_MT_FRAME_SIZE,
                        &m_width, &m_height),
                 "Read camera size"))
        break;
      opened = true;
    } while (false);

    releaseMediaFoundation(mediaType);
    releaseMediaFoundation(currentMediaType);
    releaseMediaFoundation(readerAttributes);
    releaseMediaFoundation(source);
    if (devices) {
      for (UINT32 i = 0; i < deviceCount; ++i)
        releaseMediaFoundation(devices[i]);
      CoTaskMemFree(devices);
    }
    releaseMediaFoundation(attributes);
    if (!opened) releaseMediaFoundation(m_reader);
    return opened;
  }

  bool read(cv::Mat &frame) {
    if (!m_reader) return false;
    for (;;) {
      DWORD streamIndex = 0;
      DWORD flags = 0;
      LONGLONG timestamp = 0;
      IMFSample *sample = nullptr;
      const HRESULT result = m_reader->ReadSample(
          MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags,
          &timestamp, &sample);
      if (FAILED(result)) return false;
      if (flags & MF_SOURCE_READERF_ENDOFSTREAM) return false;
      if (!sample) continue;

      IMFMediaBuffer *buffer = nullptr;
      BYTE *data = nullptr;
      DWORD length = 0;
      bool copied = false;
      if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buffer)) &&
          SUCCEEDED(buffer->Lock(&data, nullptr, &length))) {
        cv::Mat rgba(static_cast<int>(m_height), static_cast<int>(m_width),
                     CV_8UC4, data);
        cv::cvtColor(rgba, frame, cv::COLOR_BGRA2BGR);
        buffer->Unlock();
        copied = true;
      }
      releaseMediaFoundation(buffer);
      releaseMediaFoundation(sample);
      if (copied) return true;
    }
  }

private:
  IMFSourceReader *m_reader = nullptr;
  UINT32 m_width = 0;
  UINT32 m_height = 0;
  bool m_initialized = false;
};
#endif

cv::Rect faceBox(const cv::Rect &face, const cv::Size &imageSize) {
  return face & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

void drawFrame(cv::Mat &frame, const cv::Rect &box) {
  cv::rectangle(frame, box, FRAME_COLOR, 2, cv::LINE_AA);
}

void trackFaces(const cv::Mat &previousGray, const cv::Mat &currentGray,
                std::vector<cv::Rect> &faces) {
  for (auto &face : faces) {
    cv::Rect previous = face & cv::Rect(0, 0, previousGray.cols, previousGray.rows);
    if (previous.width < 12 || previous.height < 12) continue;
    const int templateWidth = previous.width * 3 / 4;
    const int templateHeight = previous.height * 3 / 4;
    cv::Rect templateRect(previous.x + (previous.width - templateWidth) / 2,
                previous.y + (previous.height - templateHeight) / 2,
                templateWidth, templateHeight);
    const int marginX = previous.width / 2;
    const int marginY = previous.height / 2;
    cv::Rect search(previous.x - marginX, previous.y - marginY,
                    previous.width + 2 * marginX, previous.height + 2 * marginY);
    search &= cv::Rect(0, 0, currentGray.cols, currentGray.rows);
    if (search.width < previous.width || search.height < previous.height) continue;

    cv::Mat result;
    cv::matchTemplate(currentGray(search), previousGray(templateRect), result,
                      cv::TM_CCOEFF_NORMED);
    double confidence = 0.0;
    cv::Point best;
    cv::minMaxLoc(result, nullptr, &confidence, nullptr, &best);
    const int maxMove = std::max(8, previous.width / 2);
    const int trackedX = search.x + best.x - (previous.width - templateWidth) / 2;
    const int trackedY = search.y + best.y - (previous.height - templateHeight) / 2;
    const int moveX = std::abs(trackedX - previous.x);
    const int moveY = std::abs(trackedY - previous.y);
    if (confidence >= 0.60 && moveX <= maxMove && moveY <= maxMove) {
      const int maxX = std::max(0, currentGray.cols - previous.width);
      const int maxY = std::max(0, currentGray.rows - previous.height);
      face.x = std::clamp(trackedX, 0, maxX);
      face.y = std::clamp(trackedY, 0, maxY);
    }
  }
}

bool loadCustomLearner(VJLearner &learner) {
  const std::vector<std::string> paths = {
      "learner.txt", "bin/learner.txt", "../learner.txt", "C:/pbl2/learner.txt"
  };
  for (const auto &p : paths) {
    if (fs::exists(p)) {
      if (loadLearnerTxt(learner, p)) {
        std::cout << "[Model] Loaded custom model from: " << p << std::endl;
        return true;
      }
    }
  }
  return false;
}

bool loadFaceCascade(cv::CascadeClassifier &cascade) {
  return cascade.load("haarcascade/haarcascade_frontalface_alt2.xml") ||
         cascade.load("../haarcascade/haarcascade_frontalface_alt2.xml") ||
         cascade.load("C:/pbl2/haarcascade/haarcascade_frontalface_alt2.xml");
}

} // namespace

const double DOWNSCALE = 0.5;

int main() {
  std::cout << "Opening webcam..." << std::endl;

#ifdef _WIN32
  MediaFoundationCamera camera;
  if (!camera.open()) {
#else
  std::cerr << "This webcam build requires Windows Media Foundation." << std::endl;
  if (true) {
#endif
    std::cerr << "Camera failed to open." << std::endl;
    return EXIT_FAILURE;
  }

  VJLearner learner;
  const bool customAvailable = loadCustomLearner(learner);
  std::unique_ptr<FaceDetector> detector;
  if (customAvailable) {
    detector = std::make_unique<FaceDetector>(std::move(learner));
  }

  cv::CascadeClassifier faceCascade;
  const bool cascadeAvailable = loadFaceCascade(faceCascade);

  if (!customAvailable && !cascadeAvailable) {
    std::cerr << "[Error] Neither learner.txt nor OpenCV Haar cascade found!" << std::endl;
    return EXIT_FAILURE;
  }

  cv::namedWindow("Live", cv::WINDOW_NORMAL);
  cv::resizeWindow("Live", 800, 600);
  std::cout << "Webcam ready. Press Q or Esc to quit." << std::endl;

  cv::Mat frame;
  cv::Mat previousGray;
  std::vector<cv::Rect> trackedFaces;
  int missedFrames = 0;
  int frameCount = 0;

  double fps = 0.0;
  auto fpsTimer = std::chrono::high_resolution_clock::now();
  int fpsCounter = 0;

  for (;;) {
    if (!camera.read(frame) || frame.empty()) {
      std::cerr << "Blank frame." << std::endl;
      break;
    }

    // Flip frame horizontally for mirror selfie effect
    cv::flip(frame, frame, 1);

    // Calculate FPS
    fpsCounter++;
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsTimer).count();
    if (elapsed >= 1000) {
      fps = (fpsCounter * 1000.0) / elapsed;
      fpsCounter = 0;
      fpsTimer = now;
    }

    cv::resize(frame, frame, cv::Size(), DOWNSCALE, DOWNSCALE);
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    if (!previousGray.empty() && !trackedFaces.empty()) {
      trackFaces(previousGray, gray, trackedFaces);
    }

    std::vector<cv::Rect> detectedFaces;
    if (++frameCount % 6 == 1 || trackedFaces.empty()) {
      std::vector<cv::Rect> cascadeFaces;

      if (cascadeAvailable) {
        cv::Mat detectionGray;
        cv::equalizeHist(gray, detectionGray);
        const int minimumFaceSize =
            std::max(20, std::min(gray.cols, gray.rows) / 28);
        faceCascade.detectMultiScale(
            detectionGray, cascadeFaces, 1.05, 5, cv::CASCADE_SCALE_IMAGE,
            cv::Size(minimumFaceSize, minimumFaceSize));
        for (const auto &face : cascadeFaces) {
          detectedFaces.push_back(faceBox(face, frame.size()));
        }
      } else if (detector) {
        const auto candidates = detector->detect(frame, 4, 5, 4.0);
        for (const auto &candidate : candidates) {
          detectedFaces.push_back(faceBox(candidate, frame.size()));
        }
      }
    }

    if (!detectedFaces.empty()) {
      trackedFaces = std::move(detectedFaces);
      missedFrames = 0;
    } else if (++missedFrames > 12) {
      trackedFaces.clear();
    }

    previousGray = gray.clone();

    for (const auto &face : trackedFaces) {
      drawFrame(frame, face);
    }

    // Overlay FPS and face count
    cv::putText(frame, cv::format("FPS: %.1f | Faces: %zu", fps, trackedFaces.size()),
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    cv::imshow("Live", frame);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }

  return 0;
}
