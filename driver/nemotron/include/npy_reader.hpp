#pragma once

#include <string>
#include <vector>

// Minimal reader for C-order float32 .npy files (version 1.0 / 2.0 headers).
// Returns false when the file is missing / malformed / not float32.
bool load_npy_float32(const std::string& path, std::vector<float>& out, std::vector<int>& shape);
