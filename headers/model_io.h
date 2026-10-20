#pragma once

#include "vjlearner.h"
#include <string>

bool saveLearnerTxt(const VJLearner &learner, const std::string &path);
bool loadLearnerTxt(VJLearner &learner, const std::string &path);
