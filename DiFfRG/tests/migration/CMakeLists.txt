function(add_migration_probe TARGET SOURCE)
  add_library(${TARGET} OBJECT EXCLUDE_FROM_ALL ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG)
endfunction()

function(add_migration_contract_test NAME TARGET EXPECT_SUCCESS EXPECTED_DIAGNOSTICS)
  get_target_property(probe_sources ${TARGET} SOURCES)
  list(GET probe_sources 0 probe_source)
  add_test(
    NAME ${NAME}
    COMMAND
      ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET_NAME=${TARGET}
      -DSOURCE_TO_TOUCH=${CMAKE_CURRENT_SOURCE_DIR}/${probe_source} -DEXPECT_SUCCESS=${EXPECT_SUCCESS}
      "-DEXPECTED_DIAGNOSTICS=${EXPECTED_DIAGNOSTICS}" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_contract.cmake)
  # "slow": each of these recompiles a translation unit against the DiFfRG headers to assert
  # which diagnostics the compiler emits, costing 10-53s apiece (worst for the ones pulling in
  # KurganovTadmor.hh). Together they were ~56% of the default suite, and RUN_SERIAL means -j
  # cannot hide it. Run with `ctest -L slow`.
  set_tests_properties(${NAME} PROPERTIES LABELS "api;migration;slow" RUN_SERIAL TRUE)
endfunction()

function(add_clean_migration_contract_test NAME TARGET FORBIDDEN_DIAGNOSTICS)
  get_target_property(probe_sources ${TARGET} SOURCES)
  list(GET probe_sources 0 probe_source)
  add_test(
    NAME ${NAME}
    COMMAND
      ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET_NAME=${TARGET}
      -DSOURCE_TO_TOUCH=${CMAKE_CURRENT_SOURCE_DIR}/${probe_source} -DEXPECT_SUCCESS=TRUE
      "-DFORBIDDEN_DIAGNOSTICS=${FORBIDDEN_DIAGNOSTICS}" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_contract.cmake)
  # "slow": each of these recompiles a translation unit against the DiFfRG headers to assert
  # which diagnostics the compiler emits, costing 10-53s apiece (worst for the ones pulling in
  # KurganovTadmor.hh). Together they were ~56% of the default suite, and RUN_SERIAL means -j
  # cannot hide it. Run with `ctest -L slow`.
  set_tests_properties(${NAME} PROPERTIES LABELS "api;migration;slow" RUN_SERIAL TRUE)
endfunction()

add_migration_probe(migration_callable_probe callable_probe.cc)
add_migration_contract_test(
  migration_callable_diagnostics migration_callable_probe TRUE
  "OutputPath(config).run_file|OutputPath(config).run_name|OutputPath(config).field_directory|OutputPath(config).root|CsvOutput(top_folder, output_name)|OutputSettings(config), active|OutputSettings(config).configuration_config|ConfigurationMesh<dim>")

add_migration_probe(migration_data_output_failure data_output_failure.cc)
add_migration_contract_test(
  migration_data_output_diagnostic migration_data_output_failure FALSE
  "Construct OutputPath, then OutputSession<Assembler>, and pass the session to the timestepper|attach_data_output(OutputFrame<dim, VectorType>&, ...)|auto fields = output.fields(); fields.attach(...)")

add_migration_probe(migration_affine_failure affine_failure.cc)
add_migration_contract_test(
  migration_affine_diagnostic migration_affine_failure FALSE
  "apply_boundary_affine_constraints|apply_affine_constraints|AffineConstraintContext")

add_migration_probe(migration_readout_failure readout_failure.cc)
add_migration_contract_test(
  migration_readout_diagnostic migration_readout_failure FALSE
  "stable readout ID: helper(id, EoM, outputter)")

add_migration_probe(migration_output_frame_csv_failure output_frame_csv_failure.cc)
add_migration_contract_test(
  migration_output_frame_csv_diagnostic migration_output_frame_csv_failure FALSE
  "auto table = output.table(name)")

add_migration_probe(migration_output_frame_lambda_failure output_frame_lambda_failure.cc)
add_migration_contract_test(
  migration_output_frame_lambda_diagnostic migration_output_frame_lambda_failure FALSE
  "TableRecord returned by output.table(name)")

add_migration_probe(migration_output_frame_fields_failure output_frame_fields_failure.cc)
add_migration_contract_test(
  migration_output_frame_fields_diagnostic migration_output_frame_fields_failure FALSE
  "auto fields = output.fields()|fields.attach(...)")

add_migration_probe(migration_default_report_port_constructor default_report_port_constructor.cc)
add_clean_migration_contract_test(
  migration_default_report_port_constructor_clean migration_default_report_port_constructor
  "deprecated-declarations")

add_migration_probe(migration_json_value_alias json_value_alias.cc)
add_migration_contract_test(
  migration_json_value_alias_diagnostic migration_json_value_alias TRUE
  "JSONValue was renamed. Use DiFfRG::ConfigTree instead")
