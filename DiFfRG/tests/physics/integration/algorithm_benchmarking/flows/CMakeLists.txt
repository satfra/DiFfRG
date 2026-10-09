set(YangMillsFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(YangMillsFlows STATIC ${YangMillsFlows_SOURCES})
target_link_libraries(YangMillsFlows DiFfRG::DiFfRG)
target_compile_options(
  YangMillsFlows PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-unused-parameter
                         -Wno-unused-variable >)
