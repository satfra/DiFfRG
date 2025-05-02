# ##############################################################################
# Tests
# ##############################################################################

add_executable(test_diagonalization test_diagonalization.cc)
setup_test(test_diagonalization)

add_executable(test_quadrature test_quadrature.cc)
setup_test(test_quadrature)

add_executable(test_matsubara test_matsubara.cc)
setup_test(test_matsubara)

add_executable(test_quadrature_provider test_quadrature_provider.cc)
setup_test(test_quadrature_provider)

add_executable(hi hi.cc)
set(TARGET hi)
    #deal_ii_setup_target(${TARGET})

   set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
   set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

   target_compile_options(${TARGET} PRIVATE -Wno-misleading-indentation)

    # Check if the target is DiFfRG
   # if(${TARGET} STREQUAL "DiFfRG")
   #   target_include_directories(${TARGET} SYSTEM PUBLIC ${autodiff_SOURCE_DIR})
   # else()
     target_link_libraries(${TARGET} autodiff::autodiff)
   # endif()

   target_link_libraries(${TARGET} GSL::gsl)
   target_link_libraries(${TARGET} Eigen3)
   target_link_libraries(${TARGET} TBB::tbb)
   target_link_libraries(${TARGET} spdlog::spdlog)
   target_link_libraries(${TARGET} ${Boost_LIBRARIES})
#target_link_libraries(hi DiFfRG::DiFfRG)
#setup_application(hi)