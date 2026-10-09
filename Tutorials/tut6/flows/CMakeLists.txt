set(ONFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_GPU/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_GPU/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_GPU/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_GPU/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_GPU/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion_GPU/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion_GPU/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion_GPU/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion_GPU/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion_GPU/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma_GPU/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma_GPU/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma_GPU/src/CT_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma_GPU/src/AD_map_points.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma_GPU/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(ONFlows STATIC ${ONFlows_SOURCES})
add_library(ONFlows_nowarn INTERFACE)
target_compile_options(ONFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(ONFlows DiFfRG::DiFfRG ONFlows_nowarn)
