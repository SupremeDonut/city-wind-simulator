#pragma once

// D3Q19 lattice constants — matches simulation/lbm_py/d3q19.py exactly

static constexpr int Q = 19;
static constexpr float CS2 = 1.0f / 3.0f;
static constexpr float TAU = 0.55f;

// Velocity directions (rest, 6 face, 12 edge)
__constant__ int EX[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int EY[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int EZ[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};

// Opposite direction index
__constant__ int OPP[Q] = { 0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,16,15,18,17};

// Weights: 1/3, 1/18 x6, 1/36 x12
__constant__ float W[Q] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
