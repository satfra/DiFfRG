set(LPA_QM_flows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(LPA_QM_flows STATIC ${LPA_QM_flows_SOURCES})
add_library(LPA_QM_flows_nowarn INTERFACE)
target_compile_options(LPA_QM_flows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(LPA_QM_flows DiFfRG::DiFfRG LPA_QM_flows_nowarn)
