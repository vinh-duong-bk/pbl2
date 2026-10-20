#pragma once

#include <string>

bool recordModelMetadata(const std::string &databasePath,
                         const std::string &modelPath,
                         int imageSize,
                         int boostAmount,
                         int learnerCount,
                         double trainingError);
