set(YangMillsFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZAcbc/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZAcbc/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZAcbc/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Zc/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Zc/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Zc/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(YangMillsFlows STATIC ${YangMillsFlows_SOURCES})
add_library(YangMillsFlows_nowarn INTERFACE)
target_compile_options(YangMillsFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(YangMillsFlows DiFfRG::DiFfRG YangMillsFlows_nowarn)
