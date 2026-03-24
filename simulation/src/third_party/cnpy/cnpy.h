// Copyright (C) 2011 Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <complex>
#include <regex>

namespace cnpy {

    struct NpyArray {
        NpyArray(const std::vector<size_t>& _shape, size_t _word_size, bool _fortran_order)
            : shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
        {
            num_vals = 1;
            for (size_t i = 0; i < shape.size(); i++) num_vals *= shape[i];
            data_holder = std::shared_ptr<std::vector<char>>(
                new std::vector<char>(num_vals * word_size));
        }

        NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals(0) {}

        template<typename T>
        T* data() {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template<typename T>
        const T* data() const {
            return reinterpret_cast<const T*>(&(*data_holder)[0]);
        }

        size_t num_bytes() const {
            return data_holder->size();
        }

        std::shared_ptr<std::vector<char>> data_holder;
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        size_t num_vals;
    };

    using npz_t = std::map<std::string, NpyArray>;

    char BigEndianTest();
    char map_type(const std::type_info& t);
    template<typename T> std::vector<char> create_npy_header(const std::vector<size_t>& shape);
    void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
    void parse_npy_header(unsigned char* buffer, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
    void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset);

    NpyArray npz_load(std::string fname, std::string varname);
    npz_t npz_load(std::string fname);

    NpyArray npy_load(std::string fname);

    template<typename T>
    std::vector<char> create_npy_header(const std::vector<size_t>& shape) {
        std::string dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += map_type(typeid(T));
        dict += std::to_string(sizeof(T));
        dict += "', 'fortran_order': False, 'shape': (";
        dict += std::to_string(shape[0]);
        for (size_t i = 1; i < shape.size(); i++) {
            dict += ", ";
            dict += std::to_string(shape[i]);
        }
        if (shape.size() == 1) dict += ",";
        dict += "), }";

        // pad with spaces so that preamble+dict is divisible by 64
        size_t preamble_len = 10; // magic(6) + version(2) + header_len(2)
        size_t padding = 64 - ((preamble_len + dict.size() + 1) % 64); // +1 for newline
        dict.insert(dict.end(), padding, ' ');
        dict += '\n';

        std::vector<char> header;
        // magic string
        header.push_back((char)0x93);
        header.push_back('N');
        header.push_back('U');
        header.push_back('M');
        header.push_back('P');
        header.push_back('Y');
        // version 1.0
        header.push_back((char)0x01);
        header.push_back((char)0x00);
        // header length (little endian uint16)
        uint16_t dict_size = (uint16_t)dict.size();
        header.push_back((char)(dict_size & 0xFF));
        header.push_back((char)((dict_size >> 8) & 0xFF));
        // dict
        header.insert(header.end(), dict.begin(), dict.end());

        return header;
    }

    template<typename T>
    void npy_save(std::string fname, const T* data, const std::vector<size_t> shape, std::string mode = "w") {
        FILE* fp = nullptr;
        std::vector<size_t> true_data_shape;

        if (mode == "a") {
#ifdef _WIN32
            fopen_s(&fp, fname.c_str(), "r+b");
#else
            fp = fopen(fname.c_str(), "r+b");
#endif
        }

        if (fp) {
            // file exists, append
            size_t word_size;
            bool fortran_order;
            parse_npy_header(fp, word_size, true_data_shape, fortran_order);
            assert(!fortran_order);

            if (word_size != sizeof(T)) {
                std::cout << "libnpy error: appending to file with word size "
                          << word_size << " but sizeof(T) is " << sizeof(T) << std::endl;
                assert(word_size == sizeof(T));
            }
            if (true_data_shape.size() != shape.size()) {
                std::cout << "libnpy error: shape dimensions don't match" << std::endl;
                assert(true_data_shape.size() == shape.size());
            }
            // check trailing dimensions match
            for (size_t i = 1; i < shape.size(); i++) {
                if (shape[i] != true_data_shape[i]) {
                    std::cout << "libnpy error: trailing dimensions don't match" << std::endl;
                    assert(shape[i] == true_data_shape[i]);
                }
            }
            true_data_shape[0] += shape[0];
        }
        else {
#ifdef _WIN32
            fopen_s(&fp, fname.c_str(), "wb");
#else
            fp = fopen(fname.c_str(), "wb");
#endif
            true_data_shape = shape;
        }

        std::vector<char> header = create_npy_header<T>(true_data_shape);

        size_t nels = 1;
        for (size_t i = 0; i < shape.size(); i++) nels *= shape[i];

        fseek(fp, 0, SEEK_SET);
        fwrite(&header[0], sizeof(char), header.size(), fp);
        fseek(fp, 0, SEEK_END);
        fwrite(data, sizeof(T), nels, fp);
        fclose(fp);
    }

    template<typename T>
    void npz_save(std::string zipname, std::string fname, const T* data,
                  const std::vector<size_t>& shape, std::string mode = "w")
    {
        // first, append a .npy to the fname
        fname += ".npy";

        // now build the .npy header + data
        std::vector<char> npy_header = create_npy_header<T>(shape);

        size_t nels = 1;
        for (size_t i = 0; i < shape.size(); i++) nels *= shape[i];
        size_t nbytes = nels * sizeof(T) + npy_header.size();

        // get the CRC of the data (uncompressed npy data)
        uint32_t crc = crc32(0L, (unsigned char*)&npy_header[0], (uInt)npy_header.size());
        crc = crc32(crc, (unsigned char*)data, (uInt)(nels * sizeof(T)));

        // build the local file header
        std::vector<char> local_header;
        // local file header signature
        local_header.push_back('P');
        local_header.push_back('K');
        local_header.push_back((char)0x03);
        local_header.push_back((char)0x04);
        // version needed to extract (20)
        local_header.push_back((char)0x14);
        local_header.push_back((char)0x00);
        // general purpose bit flag
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);
        // compression method: 0 = stored (no compression)
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);
        // last mod file time/date
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);

        // crc-32
        local_header.push_back((char)(crc & 0xFF));
        local_header.push_back((char)((crc >> 8) & 0xFF));
        local_header.push_back((char)((crc >> 16) & 0xFF));
        local_header.push_back((char)((crc >> 24) & 0xFF));

        // compressed size (same as uncompressed for stored)
        uint32_t compressed_size = (uint32_t)nbytes;
        local_header.push_back((char)(compressed_size & 0xFF));
        local_header.push_back((char)((compressed_size >> 8) & 0xFF));
        local_header.push_back((char)((compressed_size >> 16) & 0xFF));
        local_header.push_back((char)((compressed_size >> 24) & 0xFF));

        // uncompressed size
        uint32_t uncompressed_size = (uint32_t)nbytes;
        local_header.push_back((char)(uncompressed_size & 0xFF));
        local_header.push_back((char)((uncompressed_size >> 8) & 0xFF));
        local_header.push_back((char)((uncompressed_size >> 16) & 0xFF));
        local_header.push_back((char)((uncompressed_size >> 24) & 0xFF));

        // file name length
        uint16_t fname_len = (uint16_t)fname.size();
        local_header.push_back((char)(fname_len & 0xFF));
        local_header.push_back((char)((fname_len >> 8) & 0xFF));

        // extra field length
        local_header.push_back((char)0x00);
        local_header.push_back((char)0x00);

        // file name
        local_header.insert(local_header.end(), fname.begin(), fname.end());

        // now build the central directory header
        std::vector<char> central_header;
        central_header.push_back('P');
        central_header.push_back('K');
        central_header.push_back((char)0x01);
        central_header.push_back((char)0x02);
        // version made by (20)
        central_header.push_back((char)0x14);
        central_header.push_back((char)0x00);
        // version needed (20)
        central_header.push_back((char)0x14);
        central_header.push_back((char)0x00);
        // general purpose bit flag
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // compression method
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // last mod file time/date
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);

        // crc-32
        central_header.push_back((char)(crc & 0xFF));
        central_header.push_back((char)((crc >> 8) & 0xFF));
        central_header.push_back((char)((crc >> 16) & 0xFF));
        central_header.push_back((char)((crc >> 24) & 0xFF));

        // compressed size
        central_header.push_back((char)(compressed_size & 0xFF));
        central_header.push_back((char)((compressed_size >> 8) & 0xFF));
        central_header.push_back((char)((compressed_size >> 16) & 0xFF));
        central_header.push_back((char)((compressed_size >> 24) & 0xFF));

        // uncompressed size
        central_header.push_back((char)(uncompressed_size & 0xFF));
        central_header.push_back((char)((uncompressed_size >> 8) & 0xFF));
        central_header.push_back((char)((uncompressed_size >> 16) & 0xFF));
        central_header.push_back((char)((uncompressed_size >> 24) & 0xFF));

        // file name length
        central_header.push_back((char)(fname_len & 0xFF));
        central_header.push_back((char)((fname_len >> 8) & 0xFF));

        // extra field length
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // file comment length
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // disk number start
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // internal file attributes
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        // external file attributes
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);

        // relative offset of local file header - will be filled in below
        // placeholder for now
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);
        central_header.push_back((char)0x00);

        // file name
        central_header.insert(central_header.end(), fname.begin(), fname.end());

        // Now actually write the file
        FILE* fp = nullptr;
        uint16_t nrecs = 0;
        size_t global_header_offset = 0;
        std::vector<char> existing_global_header;

        if (mode == "a") {
#ifdef _WIN32
            fopen_s(&fp, zipname.c_str(), "r+b");
#else
            fp = fopen(zipname.c_str(), "r+b");
#endif
        }

        if (fp) {
            // zip file exists. read the end-of-central-directory to find existing records
            size_t global_header_size;
            parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);

            // read existing central directory
            fseek(fp, (long)global_header_offset, SEEK_SET);
            existing_global_header.resize(global_header_size);
            size_t res = fread(&existing_global_header[0], sizeof(char), global_header_size, fp);
            if (res != global_header_size) {
                throw std::runtime_error("npz_save: failed to read existing central directory");
            }

            // position to overwrite the central directory (where we'll write the new local file)
            fseek(fp, (long)global_header_offset, SEEK_SET);
        }
        else {
#ifdef _WIN32
            fopen_s(&fp, zipname.c_str(), "wb");
#else
            fp = fopen(zipname.c_str(), "wb");
#endif
        }

        if (!fp) {
            throw std::runtime_error("npz_save: unable to open " + zipname + " for writing");
        }

        // update local file header offset in central directory entry
        uint32_t local_header_offset = (uint32_t)global_header_offset;
        central_header[42] = (char)(local_header_offset & 0xFF);
        central_header[43] = (char)((local_header_offset >> 8) & 0xFF);
        central_header[44] = (char)((local_header_offset >> 16) & 0xFF);
        central_header[45] = (char)((local_header_offset >> 24) & 0xFF);

        // write local header
        fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
        // write npy header
        fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
        // write data
        fwrite(data, sizeof(T), nels, fp);

        // write all central directory records (existing + new)
        size_t new_global_header_offset = ftell(fp);
        if (!existing_global_header.empty()) {
            fwrite(&existing_global_header[0], sizeof(char), existing_global_header.size(), fp);
        }
        fwrite(&central_header[0], sizeof(char), central_header.size(), fp);
        size_t new_global_header_size = existing_global_header.size() + central_header.size();

        // write end of central directory record
        std::vector<char> footer;
        footer.push_back('P');
        footer.push_back('K');
        footer.push_back((char)0x05);
        footer.push_back((char)0x06);
        // disk number
        footer.push_back((char)0x00);
        footer.push_back((char)0x00);
        // disk number with central directory
        footer.push_back((char)0x00);
        footer.push_back((char)0x00);
        // number of records on this disk
        uint16_t new_nrecs = nrecs + 1;
        footer.push_back((char)(new_nrecs & 0xFF));
        footer.push_back((char)((new_nrecs >> 8) & 0xFF));
        // total number of records
        footer.push_back((char)(new_nrecs & 0xFF));
        footer.push_back((char)((new_nrecs >> 8) & 0xFF));
        // size of central directory
        uint32_t gh_size = (uint32_t)new_global_header_size;
        footer.push_back((char)(gh_size & 0xFF));
        footer.push_back((char)((gh_size >> 8) & 0xFF));
        footer.push_back((char)((gh_size >> 16) & 0xFF));
        footer.push_back((char)((gh_size >> 24) & 0xFF));
        // offset of start of central directory
        uint32_t gh_off = (uint32_t)new_global_header_offset;
        footer.push_back((char)(gh_off & 0xFF));
        footer.push_back((char)((gh_off >> 8) & 0xFF));
        footer.push_back((char)((gh_off >> 16) & 0xFF));
        footer.push_back((char)((gh_off >> 24) & 0xFF));
        // comment length
        footer.push_back((char)0x00);
        footer.push_back((char)0x00);

        fwrite(&footer[0], sizeof(char), footer.size(), fp);
        fclose(fp);
    }

    // convenience overload with C-style ndims parameter (size_t shape array)
    template<typename T>
    void npz_save(std::string zipname, std::string fname, const T* data,
                  const size_t* shape, size_t ndims, std::string mode = "w")
    {
        std::vector<size_t> shapevec(shape, shape + ndims);
        npz_save(zipname, fname, data, shapevec, mode);
    }

    // convenience overload with unsigned int shape array
    template<typename T>
    void npz_save(std::string zipname, std::string fname, const T* data,
                  const unsigned int* shape, unsigned int ndims, std::string mode = "w")
    {
        std::vector<size_t> shapevec(shape, shape + ndims);
        npz_save(zipname, fname, data, shapevec, mode);
    }

} // namespace cnpy

#endif
