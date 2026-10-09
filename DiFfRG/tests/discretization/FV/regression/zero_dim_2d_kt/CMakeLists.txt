# ##############################################################################
# 2D zero-dimensional KT paper regressions
# ##############################################################################

add_executable(zero_dim_2d_kt_case_v_regression case_v.cc)
target_link_libraries(zero_dim_2d_kt_case_v_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(zero_dim_2d_kt_case_v_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 720000)

add_executable(zero_dim_2d_kt_gaussian_regression gaussian.cc)
target_link_libraries(zero_dim_2d_kt_gaussian_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(zero_dim_2d_kt_gaussian_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 720000)
