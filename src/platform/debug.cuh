#pragma once
#include <cstdint>
#include <cstdio>

// Debug flag — set from host before kernel launch
__device__ static bool g_debug = false;

#define DPRINTF(...) do { if (g_debug) printf(__VA_ARGS__); } while (0)
