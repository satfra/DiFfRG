# ##############################################################################
# 2D O(2) model KT regression
# ##############################################################################

add_executable(o2_model_kt_2D_case_i_regression regression_case_i.cc)
target_link_libraries(o2_model_kt_2D_case_i_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(o2_model_kt_2D_case_i_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 3600)

add_executable(o2_model_kt_2D_case_ii_regression regression_case_ii.cc)
target_link_libraries(o2_model_kt_2D_case_ii_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(o2_model_kt_2D_case_ii_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 3600)

add_executable(o2_model_kt_2D_case_iii_regression regression_case_iii.cc)
target_link_libraries(o2_model_kt_2D_case_iii_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(o2_model_kt_2D_case_iii_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 3600)

add_executable(o2_model_kt_2D_case_iv_regression regression_case_iv.cc)
target_link_libraries(o2_model_kt_2D_case_iv_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(o2_model_kt_2D_case_iv_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 3600)
