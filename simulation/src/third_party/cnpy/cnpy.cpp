// Copyright (C) 2011 Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <regex>

namespace cnpy {

char BigEndianTest() {
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info& t) {
    if (t == typeid(float))       return 'f';
    if (t == typeid(double))      return 'f';
    if (t == typeid(long double)) return 'f';

    if (t == typeid(int))            return 'i';
    if (t == typeid(char))           return 'i';
    if (t == typeid(short))          return 'i';
    if (t == typeid(long))           return 'i';
    if (t == typeid(long long))      return 'i';
    if (t == typeid(int8_t))         return 'i';
    if (t == typeid(int16_t))        return 'i';
    if (t == typeid(int32_t))        return 'i';
    if (t == typeid(int64_t))        return 'i';

    if (t == typeid(unsigned char))      return 'u';
    if (t == typeid(unsigned short))     return 'u';
    if (t == typeid(unsigned int))       return 'u';
    if (t == typeid(unsigned long))      return 'u';
    if (t == typeid(unsigned long long)) return 'u';
    if (t == typeid(uint8_t))            return 'u';
    if (t == typeid(uint16_t))           return 'u';
    if (t == typeid(uint32_t))           return 'u';
    if (t == typeid(uint64_t))           return 'u';

    if (t == typeid(bool)) return 'b';

    if (t == typeid(std::complex<float>))  return 'c';
    if (t == typeid(std::complex<double>)) return 'c';

    throw std::runtime_error("cnpy error: unknown type");
}

static void parse_npy_header_str(std::string header, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    // fortran order
    size_t loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");

    size_t loc2 = header.find(":", loc1);
    std::string fortran_str = header.substr(loc2 + 1);
    fortran_str = fortran_str.substr(0, fortran_str.find(","));
    // trim
    fortran_str.erase(0, fortran_str.find_first_not_of(" \t\n\r"));
    fortran_str.erase(fortran_str.find_last_not_of(" \t\n\r") + 1);
    fortran_order = (fortran_str == "True");

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::string shape_str = header.substr(loc1 + 1, loc2 - loc1 - 1);
    shape.clear();

    if (!shape_str.empty()) {
        // trim
        shape_str.erase(0, shape_str.find_first_not_of(" \t\n\r"));
        shape_str.erase(shape_str.find_last_not_of(" \t\n\r") + 1);

        if (!shape_str.empty()) {
            std::istringstream iss(shape_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty()) {
                    shape.push_back((size_t)std::stoull(token));
                }
            }
        }
    }

    // endianness, word size, data type
    // look for descr
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");

    // Header format: 'descr': '<f4'  — need to find the value between the
    // 3rd and 4th single quotes after "descr" (skip key-closing and value-opening quotes)
    loc2 = header.find("'", loc1 + 1);           // closing ' of key 'descr'
    size_t loc3 = header.find("'", loc2 + 1);    // opening ' of value
    size_t loc4 = header.find("'", loc3 + 1);    // closing ' of value
    std::string dtype_str = header.substr(loc3 + 1, loc4 - loc3 - 1);

    // dtype_str is something like "<f4" or "|u1"
    // the last characters are the word size
    std::string word_size_str;
    for (size_t i = dtype_str.size() - 1; i > 0; i--) {
        if (dtype_str[i] >= '0' && dtype_str[i] <= '9') {
            word_size_str = dtype_str[i] + word_size_str;
        } else {
            break;
        }
    }
    word_size = (size_t)std::stoull(word_size_str);
}

void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 8, fp);
    if (res != 8)
        throw std::runtime_error("parse_npy_header: failed to read npy header magic");

    // check magic string
    if (buffer[0] != (char)0x93 || buffer[1] != 'N' || buffer[2] != 'U' ||
        buffer[3] != 'M' || buffer[4] != 'P' || buffer[5] != 'Y')
        throw std::runtime_error("parse_npy_header: invalid npy magic string");

    uint8_t major_version = buffer[6];
    // uint8_t minor_version = buffer[7]; // unused

    uint32_t header_len = 0;
    if (major_version == 1) {
        uint16_t hlen;
        res = fread(&hlen, sizeof(uint16_t), 1, fp);
        if (res != 1)
            throw std::runtime_error("parse_npy_header: failed to read header length");
        header_len = hlen;
    }
    else if (major_version == 2) {
        res = fread(&header_len, sizeof(uint32_t), 1, fp);
        if (res != 1)
            throw std::runtime_error("parse_npy_header: failed to read header length");
    }
    else {
        throw std::runtime_error("parse_npy_header: unsupported npy version");
    }

    std::vector<char> header_data(header_len);
    res = fread(&header_data[0], sizeof(char), header_len, fp);
    if (res != header_len)
        throw std::runtime_error("parse_npy_header: failed to read header data");

    std::string header(header_data.begin(), header_data.end());
    parse_npy_header_str(header, word_size, shape, fortran_order);
}

void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    // check magic
    if (buffer[0] != (unsigned char)0x93 || buffer[1] != 'N' || buffer[2] != 'U' ||
        buffer[3] != 'M' || buffer[4] != 'P' || buffer[5] != 'Y')
        throw std::runtime_error("parse_npy_header: invalid npy magic string");

    uint8_t major_version = buffer[6];
    // uint8_t minor_version = buffer[7]; // unused

    uint32_t header_len = 0;
    size_t offset = 8;
    if (major_version == 1) {
        uint16_t hlen;
        memcpy(&hlen, buffer + offset, sizeof(uint16_t));
        header_len = hlen;
        offset += 2;
    }
    else if (major_version == 2) {
        memcpy(&header_len, buffer + offset, sizeof(uint32_t));
        offset += 4;
    }
    else {
        throw std::runtime_error("parse_npy_header: unsupported npy version");
    }

    std::string header((char*)buffer + offset, header_len);
    parse_npy_header_str(header, word_size, shape, fortran_order);
}

void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset) {
    // end of central directory record is at least 22 bytes
    // search backwards for the signature PK\x05\x06
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22)
        throw std::runtime_error("parse_zip_footer: failed to read footer");

    if (footer[0] != 'P' || footer[1] != 'K' || footer[2] != (char)0x05 || footer[3] != (char)0x06)
        throw std::runtime_error("parse_zip_footer: invalid end of central directory signature");

    uint16_t disk_nrecs;
    memcpy(&disk_nrecs, &footer[8], sizeof(uint16_t));
    nrecs = disk_nrecs;

    uint32_t gh_size;
    memcpy(&gh_size, &footer[12], sizeof(uint32_t));
    global_header_size = gh_size;

    uint32_t gh_offset;
    memcpy(&gh_offset, &footer[16], sizeof(uint32_t));
    global_header_offset = gh_offset;
}

NpyArray npy_load(std::string fname) {
    FILE* fp = nullptr;
#ifdef _WIN32
    fopen_s(&fp, fname.c_str(), "rb");
#else
    fp = fopen(fname.c_str(), "rb");
#endif

    if (!fp) throw std::runtime_error("npy_load: unable to open file " + fname);

    size_t word_size;
    std::vector<size_t> shape;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("npy_load: failed to read data");
    fclose(fp);
    return arr;
}

npz_t npz_load(std::string fname) {
    FILE* fp = nullptr;
#ifdef _WIN32
    fopen_s(&fp, fname.c_str(), "rb");
#else
    fp = fopen(fname.c_str(), "rb");
#endif

    if (!fp) throw std::runtime_error("npz_load: unable to open file " + fname);

    npz_t arrays;

    while (true) {
        // read local file header
        char local_header[30];
        size_t headerres = fread(local_header, sizeof(char), 30, fp);
        if (headerres != 30) break; // end of file

        // check signature
        if (local_header[0] != 'P' || local_header[1] != 'K') break;
        // check if this is a local file header (03 04) vs central dir (01 02)
        if (local_header[2] == 0x01 && local_header[3] == 0x02) break; // central directory, we're done
        if (local_header[2] == 0x05 && local_header[3] == 0x06) break; // end of central directory
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // compression method
        uint16_t compr_method;
        memcpy(&compr_method, &local_header[8], sizeof(uint16_t));

        // compressed size
        uint32_t compr_bytes;
        memcpy(&compr_bytes, &local_header[18], sizeof(uint32_t));

        // uncompressed size
        uint32_t uncompr_bytes;
        memcpy(&uncompr_bytes, &local_header[22], sizeof(uint32_t));

        // file name length
        uint16_t fname_len;
        memcpy(&fname_len, &local_header[26], sizeof(uint16_t));

        // extra field length
        uint16_t extra_len;
        memcpy(&extra_len, &local_header[28], sizeof(uint16_t));

        // read file name
        std::vector<char> vname(fname_len);
        size_t nameres = fread(&vname[0], sizeof(char), fname_len, fp);
        if (nameres != fname_len) break;
        std::string varname(vname.begin(), vname.end());

        // strip .npy extension
        if (varname.size() >= 4 && varname.substr(varname.size() - 4) == ".npy") {
            varname = varname.substr(0, varname.size() - 4);
        }

        // skip extra field
        if (extra_len > 0) {
            fseek(fp, extra_len, SEEK_CUR);
        }

        // read data
        if (compr_method == 0) {
            // stored (no compression)
            // the data is a raw .npy file
            // parse the npy header from the data
            size_t data_start = ftell(fp);

            size_t word_size;
            std::vector<size_t> shape;
            bool fortran_order;
            parse_npy_header(fp, word_size, shape, fortran_order);

            NpyArray arr(shape, word_size, fortran_order);
            size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
            if (nread != arr.num_bytes()) {
                throw std::runtime_error("npz_load: failed to read array data for " + varname);
            }

            arrays[varname] = arr;
        }
        else if (compr_method == 8) {
            // deflate
            std::vector<unsigned char> compr_data(compr_bytes);
            size_t cres = fread(&compr_data[0], sizeof(unsigned char), compr_bytes, fp);
            if (cres != compr_bytes) {
                throw std::runtime_error("npz_load: failed to read compressed data for " + varname);
            }

            std::vector<unsigned char> uncompr_data(uncompr_bytes);

            z_stream d_stream;
            d_stream.zalloc = Z_NULL;
            d_stream.zfree = Z_NULL;
            d_stream.opaque = Z_NULL;
            d_stream.avail_in = compr_bytes;
            d_stream.next_in = &compr_data[0];
            d_stream.avail_out = uncompr_bytes;
            d_stream.next_out = &uncompr_data[0];

            int err = inflateInit2(&d_stream, -MAX_WBITS);
            if (err != Z_OK) throw std::runtime_error("npz_load: inflateInit2 failed");

            err = inflate(&d_stream, Z_FINISH);
            if (err != Z_STREAM_END) {
                inflateEnd(&d_stream);
                throw std::runtime_error("npz_load: inflate failed");
            }
            inflateEnd(&d_stream);

            // now parse the uncompressed npy data
            size_t word_size;
            std::vector<size_t> shape;
            bool fortran_order;
            parse_npy_header(&uncompr_data[0], word_size, shape, fortran_order);

            NpyArray arr(shape, word_size, fortran_order);

            // figure out where the data starts in the uncompressed buffer
            // magic(6) + version(2) + header_len_field + header_data
            uint8_t major_version = uncompr_data[6];
            size_t header_offset = 8;
            uint32_t header_len = 0;
            if (major_version == 1) {
                uint16_t hlen;
                memcpy(&hlen, &uncompr_data[8], sizeof(uint16_t));
                header_len = hlen;
                header_offset = 10;
            }
            else {
                memcpy(&header_len, &uncompr_data[8], sizeof(uint32_t));
                header_offset = 12;
            }
            size_t data_offset = header_offset + header_len;

            memcpy(arr.data<char>(), &uncompr_data[data_offset], arr.num_bytes());
            arrays[varname] = arr;
        }
        else {
            throw std::runtime_error("npz_load: unsupported compression method");
        }
    }

    fclose(fp);
    return arrays;
}

NpyArray npz_load(std::string fname, std::string varname) {
    npz_t arrays = npz_load(fname);
    auto it = arrays.find(varname);
    if (it == arrays.end()) {
        throw std::runtime_error("npz_load: variable '" + varname + "' not found in " + fname);
    }
    return it->second;
}

} // namespace cnpy
