#include "npy_reader.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

bool load_npy_float32(const std::string& path, std::vector<float>& out, std::vector<int>& shape) {
    std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize size = file.tellg();
    file.seekg(0);
    if (size < 10) return false;

    char magic[6];
    file.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;

    unsigned char version[2];
    file.read(reinterpret_cast<char*>(version), 2);
    std::uint32_t header_len = 0;
    if (version[0] == 1) {
        unsigned char raw[2];
        file.read(reinterpret_cast<char*>(raw), 2);
        header_len = static_cast<std::uint32_t>(raw[0]) | (static_cast<std::uint32_t>(raw[1]) << 8);
    } else {
        unsigned char raw[4];
        file.read(reinterpret_cast<char*>(raw), 4);
        header_len = static_cast<std::uint32_t>(raw[0]) | (static_cast<std::uint32_t>(raw[1]) << 8) |
                     (static_cast<std::uint32_t>(raw[2]) << 16) | (static_cast<std::uint32_t>(raw[3]) << 24);
    }

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    if (!file) return false;

    if (header.find("'<f4'") == std::string::npos && header.find("\"<f4\"") == std::string::npos) return false;
    if (header.find("False") == std::string::npos) return false;  // fortran_order must be False

    shape.clear();
    std::size_t elements = 1;
    const std::size_t shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) return false;
    const std::size_t open = header.find('(', shape_pos);
    const std::size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) return false;
    std::string dims = header.substr(open + 1, close - open - 1);
    std::stringstream stream(dims);
    std::string item;
    while (std::getline(stream, item, ',')) {
        std::size_t begin = item.find_first_not_of(" \t");
        if (begin == std::string::npos) continue;
        std::size_t end = item.find_last_not_of(" \t");
        const int dim = std::atoi(item.substr(begin, end - begin + 1).c_str());
        if (dim < 0) return false;
        shape.push_back(dim);
        elements *= static_cast<std::size_t>(dim);
    }
    if (shape.empty()) return false;

    const std::streamsize data_bytes = static_cast<std::streamsize>(elements) * sizeof(float);
    if (size - file.tellg() < data_bytes) return false;

    out.resize(elements);
    file.read(reinterpret_cast<char*>(out.data()), data_bytes);
    return static_cast<bool>(file);
}
