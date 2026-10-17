set(QMBenchFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/MesonsPlain/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MesonsPlain/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MesonsPlain/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/VAuto/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/VAuto/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/VAuto/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/VAuto/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/VAuto/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZPi/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZPi/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZPi/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZSig/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZSig/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZSig/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

find_package(NumTracer REQUIRED HINTS ~/.local/share/NumTracer)

add_library(QMBenchFlows STATIC ${QMBenchFlows_SOURCES})
add_library(QMBenchFlows_nowarn INTERFACE)
target_compile_options(QMBenchFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(QMBenchFlows DiFfRG::DiFfRG QMBenchFlows_nowarn NumTracer::NumTracer)

numtracer_add_numtrace(QMBenchFlows ${CMAKE_CURRENT_SOURCE_DIR})