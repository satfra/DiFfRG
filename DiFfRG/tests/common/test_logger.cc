#include <catch2/catch_all.hpp>

#include <DiFfRG/common/utils.hh>

TEST_CASE("build_logger functionality", "[spdlog_utils]")
{
  const std::string test_logger_name_base = "test_logger_";
  const std::string test_log_filename_base = "test_log_";
  static int test_counter = 0;
  test_counter++; // Ensure unique names for each run/section if tests run sequentially without full drop

  std::string logger_name = test_logger_name_base + std::to_string(test_counter);
  std::string filename = test_log_filename_base + std::to_string(test_counter) + ".txt";

  // Ensure the logger does not exist initially by trying to drop it (if it somehow persisted)
  // and then checking with spdlog::get()
  spdlog::drop(logger_name); // remove if present from previous failed test
  REQUIRE(spdlog::get(logger_name) == nullptr);

  SECTION("Logger creation when it does not exist")
  {
    INFO("Testing creation for: " << logger_name);
    auto logger = DiFfRG::build_logger(logger_name, filename);

    REQUIRE(logger != nullptr);
    REQUIRE(logger->name() == logger_name);

    // Verify it's now in the registry
    auto retrieved_logger = spdlog::get(logger_name);
    REQUIRE(retrieved_logger != nullptr);
    REQUIRE(retrieved_logger == logger); // Should be the same shared_ptr instance

    // Cleanup for this section
    logger->flush(); // Ensure file is written before attempting to remove
    spdlog::drop(logger_name);
    // remove_log_file(filename);
    REQUIRE(spdlog::get(logger_name) == nullptr); // Verify cleanup
  }

  // Use a different name for the next section to ensure independence,
  // or rely on the previous section's cleanup.
  // For robust tests, unique names or guaranteed cleanup per section is best.
  std::string logger_name_retrieve = test_logger_name_base + "retrieve_" + std::to_string(test_counter);
  std::string filename_retrieve = test_log_filename_base + "retrieve_" + std::to_string(test_counter) + ".txt";
  spdlog::drop(logger_name_retrieve); // Ensure clean slate

  SECTION("Logger retrieval when it already exists")
  {
    INFO("Testing retrieval for: " << logger_name_retrieve);
    // First, create the logger
    auto initial_logger = DiFfRG::build_logger(logger_name_retrieve, filename_retrieve);
    REQUIRE(initial_logger != nullptr);
    REQUIRE(initial_logger->name() == logger_name_retrieve);

    // Now, try to get/create it again
    auto retrieved_logger = DiFfRG::build_logger(logger_name_retrieve, filename_retrieve);
    REQUIRE(retrieved_logger != nullptr);
    REQUIRE(retrieved_logger->name() == logger_name_retrieve);

    // Crucially, check they are the same instance
    REQUIRE(retrieved_logger == initial_logger);

    // Check that calling spdlog::get directly also returns the same instance
    auto direct_get_logger = spdlog::get(logger_name_retrieve);
    REQUIRE(direct_get_logger != nullptr);
    REQUIRE(direct_get_logger == initial_logger);

    // Cleanup for this section
    initial_logger->flush();
    spdlog::drop(logger_name_retrieve);
    // remove_log_file(filename_retrieve);
    REQUIRE(spdlog::get(logger_name_retrieve) == nullptr);
  }

  SECTION("Calling build_logger multiple times returns the same instance")
  {
    std::string logger_name_multi = test_logger_name_base + "multi_" + std::to_string(test_counter);
    std::string filename_multi = test_log_filename_base + "multi_" + std::to_string(test_counter) + ".txt";
    spdlog::drop(logger_name_multi); // Ensure clean slate

    INFO("Testing multiple calls for: " << logger_name_multi);
    auto logger1 = DiFfRG::build_logger(logger_name_multi, filename_multi);
    REQUIRE(logger1 != nullptr);

    auto logger2 = DiFfRG::build_logger(logger_name_multi, filename_multi);
    REQUIRE(logger2 != nullptr);
    REQUIRE(logger1 == logger2);

    auto logger3 = spdlog::get(logger_name_multi); // get directly
    REQUIRE(logger3 != nullptr);
    REQUIRE(logger1 == logger3);

    // Cleanup
    logger1->flush();
    spdlog::drop(logger_name_multi);
    // remove_log_file(filename_multi);
    REQUIRE(spdlog::get(logger_name_multi) == nullptr);
  }
}
