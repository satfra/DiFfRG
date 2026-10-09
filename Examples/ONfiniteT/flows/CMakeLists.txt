set(ONFiniteTFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_pion/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V_sigma/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(ONFiniteTFlows STATIC ${ONFiniteTFlows_SOURCES})
add_library(ONFiniteTFlows_nowarn INTERFACE)
target_compile_options(ONFiniteTFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(ONFiniteTFlows DiFfRG::DiFfRG ONFiniteTFlows_nowarn)
