cmake_minimum_required(VERSION 3.16)

project(arb_migration_c_chassis C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(double_interval_ref SHARED
    src/double_interval_ref.c
)

target_include_directories(double_interval_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(double_interval_ref PRIVATE DI_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(double_interval_ref PRIVATE /W4)
else ()
    target_compile_options(double_interval_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(double_interval_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_library(hypgeom_ref SHARED
    src/hypgeom_ref.c
)

target_include_directories(hypgeom_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(hypgeom_ref PRIVATE double_interval_ref)
target_compile_definitions(hypgeom_ref PRIVATE HYPGEOM_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(hypgeom_ref PRIVATE /W4)
else ()
    target_compile_options(hypgeom_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(hypgeom_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_library(arb_core_ref SHARED
    src/arb_core_ref.c
)

target_include_directories(arb_core_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arb_core_ref PRIVATE double_interval_ref)
target_compile_definitions(arb_core_ref PRIVATE ARB_CORE_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_core_ref PRIVATE /W4)
else ()
    target_compile_options(arb_core_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_core_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_library(acb_core_ref SHARED
    src/acb_core_ref.c
)

target_include_directories(acb_core_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_core_ref PRIVATE double_interval_ref)
target_compile_definitions(acb_core_ref PRIVATE ACB_CORE_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_core_ref PRIVATE /W4)
else ()
    target_compile_options(acb_core_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_core_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_library(dft_ref SHARED
    src/dft_ref.c
)

target_include_directories(dft_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dft_ref PRIVATE double_interval_ref)
target_compile_definitions(dft_ref PRIVATE DFT_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(dft_ref PRIVATE /W4)
else ()
    target_compile_options(dft_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(dft_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

add_library(acb_calc_ref SHARED
    src/acb_calc_ref.c
)

target_include_directories(acb_calc_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_calc_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_calc_ref PRIVATE ACB_CALC_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_calc_ref PRIVATE /W4)
else ()
    target_compile_options(acb_calc_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_calc_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acb_dirichlet_ref SHARED
    src/acb_dirichlet_ref.c
)

target_include_directories(acb_dirichlet_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_dirichlet_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_dirichlet_ref PRIVATE ACB_DIRICHLET_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_dirichlet_ref PRIVATE /W4)
else ()
    target_compile_options(acb_dirichlet_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_dirichlet_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acb_elliptic_ref SHARED
    src/acb_elliptic_ref.c
)

target_include_directories(acb_elliptic_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_elliptic_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_elliptic_ref PRIVATE ACB_ELLIPTIC_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_elliptic_ref PRIVATE /W4)
else ()
    target_compile_options(acb_elliptic_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_elliptic_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acb_mat_ref SHARED
    src/acb_mat_ref.c
)

target_include_directories(acb_mat_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_mat_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_mat_ref PRIVATE ACB_MAT_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_mat_ref PRIVATE /W4)
else ()
    target_compile_options(acb_mat_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_mat_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acb_modular_ref SHARED
    src/acb_modular_ref.c
)

target_include_directories(acb_modular_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_modular_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_modular_ref PRIVATE ACB_MODULAR_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_modular_ref PRIVATE /W4)
else ()
    target_compile_options(acb_modular_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_modular_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acb_poly_ref SHARED
    src/acb_poly_ref.c
)

target_include_directories(acb_poly_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(acb_poly_ref PRIVATE double_interval_ref)

target_compile_definitions(acb_poly_ref PRIVATE ACB_POLY_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acb_poly_ref PRIVATE /W4)
else ()
    target_compile_options(acb_poly_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acb_poly_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(acf_ref SHARED
    src/acf_ref.c
)

target_include_directories(acf_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(acf_ref PRIVATE ACF_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(acf_ref PRIVATE /W4)
else ()
    target_compile_options(acf_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(acf_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arb_calc_ref SHARED
    src/arb_calc_ref.c
)

target_include_directories(arb_calc_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arb_calc_ref PRIVATE double_interval_ref)

target_compile_definitions(arb_calc_ref PRIVATE ARB_CALC_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_calc_ref PRIVATE /W4)
else ()
    target_compile_options(arb_calc_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_calc_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arb_fmpz_poly_ref SHARED
    src/arb_fmpz_poly_ref.c
)

target_include_directories(arb_fmpz_poly_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arb_fmpz_poly_ref PRIVATE double_interval_ref)

target_compile_definitions(arb_fmpz_poly_ref PRIVATE ARB_FMPZ_POLY_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_fmpz_poly_ref PRIVATE /W4)
else ()
    target_compile_options(arb_fmpz_poly_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_fmpz_poly_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arb_fpwrap_ref SHARED
    src/arb_fpwrap_ref.c
)

target_include_directories(arb_fpwrap_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(arb_fpwrap_ref PRIVATE ARB_FPWRAP_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_fpwrap_ref PRIVATE /W4)
else ()
    target_compile_options(arb_fpwrap_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_fpwrap_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arb_mat_ref SHARED
    src/arb_mat_ref.c
)

target_include_directories(arb_mat_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arb_mat_ref PRIVATE double_interval_ref)

target_compile_definitions(arb_mat_ref PRIVATE ARB_MAT_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_mat_ref PRIVATE /W4)
else ()
    target_compile_options(arb_mat_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_mat_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arb_poly_ref SHARED
    src/arb_poly_ref.c
)

target_include_directories(arb_poly_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arb_poly_ref PRIVATE double_interval_ref)

target_compile_definitions(arb_poly_ref PRIVATE ARB_POLY_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arb_poly_ref PRIVATE /W4)
else ()
    target_compile_options(arb_poly_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arb_poly_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(arf_ref SHARED
    src/arf_ref.c
)

target_include_directories(arf_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(arf_ref PRIVATE ARF_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(arf_ref PRIVATE /W4)
else ()
    target_compile_options(arf_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(arf_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(bernoulli_ref SHARED
    src/bernoulli_ref.c
)

target_include_directories(bernoulli_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(bernoulli_ref PRIVATE BERNOULLI_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(bernoulli_ref PRIVATE /W4)
else ()
    target_compile_options(bernoulli_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(bernoulli_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(bool_mat_ref SHARED
    src/bool_mat_ref.c
)

target_include_directories(bool_mat_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(bool_mat_ref PRIVATE BOOL_MAT_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(bool_mat_ref PRIVATE /W4)
else ()
    target_compile_options(bool_mat_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(bool_mat_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(dirichlet_ref SHARED
    src/dirichlet_ref.c
)

target_include_directories(dirichlet_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dirichlet_ref PRIVATE double_interval_ref)

target_compile_definitions(dirichlet_ref PRIVATE DIRICHLET_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(dirichlet_ref PRIVATE /W4)
else ()
    target_compile_options(dirichlet_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(dirichlet_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(dlog_ref SHARED
    src/dlog_ref.c
)

target_include_directories(dlog_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(dlog_ref PRIVATE DLOG_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(dlog_ref PRIVATE /W4)
else ()
    target_compile_options(dlog_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(dlog_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(fmpr_ref SHARED
    src/fmpr_ref.c
)

target_include_directories(fmpr_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(fmpr_ref PRIVATE FMPR_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(fmpr_ref PRIVATE /W4)
else ()
    target_compile_options(fmpr_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(fmpr_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(fmpz_extras_ref SHARED
    src/fmpz_extras_ref.c
)

target_include_directories(fmpz_extras_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(fmpz_extras_ref PRIVATE FMPZ_EXTRAS_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(fmpz_extras_ref PRIVATE /W4)
else ()
    target_compile_options(fmpz_extras_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(fmpz_extras_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(fmpzi_ref SHARED
    src/fmpzi_ref.c
)

target_include_directories(fmpzi_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(fmpzi_ref PRIVATE FMPZI_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(fmpzi_ref PRIVATE /W4)
else ()
    target_compile_options(fmpzi_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(fmpzi_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(mag_ref SHARED
    src/mag_ref.c
)

target_include_directories(mag_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(mag_ref PRIVATE MAG_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(mag_ref PRIVATE /W4)
else ()
    target_compile_options(mag_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(mag_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(partitions_ref SHARED
    src/partitions_ref.c
)

target_include_directories(partitions_ref PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(partitions_ref PRIVATE PARTITIONS_REF_BUILD_DLL)

if (MSVC)
    target_compile_options(partitions_ref PRIVATE /W4)
else ()
    target_compile_options(partitions_ref PRIVATE -Wall -Wextra -Wpedantic)
endif ()

set_target_properties(partitions_ref PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
