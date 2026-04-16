/*
 * S-MAS: C API Export Layer (Task 1.7)
 * extern "C" functions for C# P/Invoke and Python ctypes.
 *
 * Usage:
 *   void*  eng = smas_create("path/to/preprocessed-data", 42);
 *   smas_init(eng);
 *   smas_reset(eng);
 *   StatePacket s = smas_step(eng, action);
 *   smas_destroy(eng);
 */
#pragma once
#include "contracts.h"

#ifdef _WIN32
  #define SMAS_API __declspec(dllexport)
#else
  #define SMAS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Create a simulation engine. Returns opaque handle.
SMAS_API void* smas_create(const char* data_dir, unsigned long long seed);

// Initialise (load data). Returns 0 on success, -1 on failure.
SMAS_API int smas_init(void* engine);

// Reset to initial conditions.
SMAS_API void smas_reset(void* engine);

// Advance one time step. Fills `out_state`.
SMAS_API void smas_step(void* engine,
                        const smas::ActionPacket* action,
                        smas::StatePacket* out_state);

// Query whether the episode is done.
SMAS_API int smas_is_done(void* engine);

// Destroy and free memory.
SMAS_API void smas_destroy(void* engine);

// Utility: return the size of StatePacket / ActionPacket for ABI checks.
SMAS_API int smas_state_packet_size(void);
SMAS_API int smas_action_packet_size(void);

#ifdef __cplusplus
}
#endif
