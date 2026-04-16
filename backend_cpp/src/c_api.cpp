/*
 * S-MAS: C API Export Layer — Implementation (Task 1.7)
 */
#include "c_api.h"
#include "simulation_engine.h"
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

SMAS_API void* smas_create(const char* data_dir, unsigned long long seed) {
    smas::EngineConfig cfg;
    cfg.data_dir = data_dir;
    cfg.seed     = seed;
    return new smas::SimulationEngine(cfg);
}

SMAS_API int smas_init(void* engine) {
    auto* eng = static_cast<smas::SimulationEngine*>(engine);
    return eng->init() ? 0 : -1;
}

SMAS_API void smas_reset(void* engine) {
    auto* eng = static_cast<smas::SimulationEngine*>(engine);
    eng->reset();
}

SMAS_API void smas_step(void* engine,
                        const smas::ActionPacket* action,
                        smas::StatePacket* out_state) {
    auto* eng = static_cast<smas::SimulationEngine*>(engine);
    smas::StatePacket s = eng->step(*action);
    std::memcpy(out_state, &s, sizeof(smas::StatePacket));
}

SMAS_API int smas_is_done(void* engine) {
    auto* eng = static_cast<smas::SimulationEngine*>(engine);
    return eng->is_done() ? 1 : 0;
}

SMAS_API void smas_destroy(void* engine) {
    auto* eng = static_cast<smas::SimulationEngine*>(engine);
    delete eng;
}

SMAS_API int smas_state_packet_size(void) {
    return static_cast<int>(sizeof(smas::StatePacket));
}

SMAS_API int smas_action_packet_size(void) {
    return static_cast<int>(sizeof(smas::ActionPacket));
}

#ifdef __cplusplus
}
#endif
