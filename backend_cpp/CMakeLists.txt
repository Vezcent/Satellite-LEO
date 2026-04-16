# ═══════════════════════════════════════════════════════════════════
#  S-MAS: C++ Physics Core — Build Configuration
#  Targets: smas_engine (shared lib) + smas_test (executable)
# ═══════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.16)
project(smas_engine VERSION 1.0 LANGUAGES CXX)

# ── C++14 (portable; C++17 features shimmed in types.h) ────────────
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Include directory ──────────────────────────────────────────────
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Source files ───────────────────────────────────────────────────
set(ENGINE_SOURCES
    src/data_loader.cpp
    src/atmosphere.cpp
    src/geometry.cpp
    src/orbital_mechanics.cpp
    src/satellite_bus.cpp
    src/stochastic.cpp
    src/simulation_engine.cpp
    src/c_api.cpp
)

# ── Shared Library (.dll / .so) ────────────────────────────────────
add_library(smas_engine SHARED ${ENGINE_SOURCES})
target_include_directories(smas_engine PUBLIC ${INCLUDE_DIR})
target_compile_definitions(smas_engine PRIVATE SMAS_BUILD_DLL)

# Platform-specific: export all symbols on Windows
if(WIN32)
    set_target_properties(smas_engine PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ── Static Library (for linking the test executable) ───────────────
add_library(smas_engine_static STATIC ${ENGINE_SOURCES})
target_include_directories(smas_engine_static PUBLIC ${INCLUDE_DIR})

# ── Test Executable ────────────────────────────────────────────────
add_executable(smas_test src/main.cpp)
target_link_libraries(smas_test PRIVATE smas_engine_static)
target_include_directories(smas_test PRIVATE ${INCLUDE_DIR})

# ── Compiler warnings ─────────────────────────────────────────────
if(MSVC)
    target_compile_options(smas_engine PRIVATE /W4)
    target_compile_options(smas_test PRIVATE /W4)
else()
    target_compile_options(smas_engine PRIVATE -Wall -Wextra -O2)
    target_compile_options(smas_test PRIVATE -Wall -Wextra -O2)
endif()

# ── Install ────────────────────────────────────────────────────────
install(TARGETS smas_engine DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# ── Print summary ──────────────────────────────────────────────────
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  S-MAS Physics Engine Build")
message(STATUS "  Shared lib: smas_engine (.dll/.so)")
message(STATUS "  Test binary: smas_test")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")
