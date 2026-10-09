execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${SOURCE_TO_TOUCH})

execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR} --target ${TARGET_NAME} --parallel 1
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_stdout
  ERROR_VARIABLE build_stderr)

set(build_log "${build_stdout}\n${build_stderr}")

# Compilers hard-wrap diagnostics (this toolchain breaks at ~72 columns, mid-identifier), so a
# raw substring search fails for any expected message that straddles a wrap point -- regardless
# of whether the compiler actually said it. Match on a whitespace-normalised copy instead.
string(REGEX REPLACE "[ \t\r\n]+" " " normalized_log "${build_log}")
if(EXPECT_SUCCESS AND NOT build_result EQUAL 0)
  message(FATAL_ERROR "${TARGET_NAME} unexpectedly failed to compile:\n${build_log}")
endif()
if(NOT EXPECT_SUCCESS AND build_result EQUAL 0)
  message(FATAL_ERROR "${TARGET_NAME} unexpectedly compiled successfully")
endif()

string(REPLACE "|" ";" expected_messages "${EXPECTED_DIAGNOSTICS}")
foreach(expected_message IN LISTS expected_messages)
  string(REGEX REPLACE "[ \t\r\n]+" " " normalized_expected "${expected_message}")
  string(FIND "${normalized_log}" "${normalized_expected}" diagnostic_position)
  if(diagnostic_position EQUAL -1)
    message(FATAL_ERROR "${TARGET_NAME} did not emit '${expected_message}':\n${build_log}")
  endif()
endforeach()

string(REPLACE "|" ";" forbidden_messages "${FORBIDDEN_DIAGNOSTICS}")
foreach(forbidden_message IN LISTS forbidden_messages)
  if(forbidden_message STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE "[ \t\r\n]+" " " normalized_forbidden "${forbidden_message}")
  string(FIND "${normalized_log}" "${normalized_forbidden}" diagnostic_position)
  if(NOT diagnostic_position EQUAL -1)
    message(FATAL_ERROR "${TARGET_NAME} unexpectedly emitted '${forbidden_message}':\n${build_log}")
  endif()
endforeach()
