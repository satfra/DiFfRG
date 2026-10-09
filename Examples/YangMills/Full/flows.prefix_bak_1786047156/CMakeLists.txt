set(YangMillsFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA3/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4SP/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4SP/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4SP/src/CT_map_1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4tadpole/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4tadpole/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ZA4tadpole/src/CT_map_1.cc
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

set_target_properties(
  YangMillsFlows
  PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE GROUP)

foreach(src IN LISTS YangMillsFlows_SOURCES)
  if(src MATCHES "/src/CT_map_[0-9]+\.cc$")
    set_source_files_properties("${src}" PROPERTIES UNITY_GROUP "all_ct_map")
  else()
    set_source_files_properties("${src}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  endif()
endforeach()
