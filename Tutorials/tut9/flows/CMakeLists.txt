set(QMFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesons/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesons/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesons/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

find_package(NumTracer REQUIRED HINTS ~/.local/share/NumTracer)

add_library(QMFlows STATIC ${QMFlows_SOURCES})
add_library(QMFlows_nowarn INTERFACE)
target_compile_options(QMFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(QMFlows DiFfRG::DiFfRG QMFlows_nowarn NumTracer::NumTracer)

numtracer_add_numtrace(QMFlows ${CMAKE_CURRENT_SOURCE_DIR})