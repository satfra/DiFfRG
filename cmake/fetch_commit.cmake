# ##############################################################################
# Shallow-fetch a single git commit (depth 1) into DEST.
#
# Used by the top-level build for dependencies pinned to a specific (non-tip)
# commit. CMake's ExternalProject GIT_SHALLOW cannot reach such commits - it only
# shallow-fetches branch tips (`clone --depth 1 --no-single-branch` then
# `checkout <sha>`, which fails with "unable to read tree"). Driving git directly
# with `git fetch --depth 1 origin <sha>` works against GitHub for any reachable
# commit and avoids downloading the full repository history.
#
# Expected -D variables: GIT_EXECUTABLE, REPO, TAG (commit SHA), DEST
# ##############################################################################

if(NOT GIT_EXECUTABLE
   OR NOT REPO
   OR NOT TAG
   OR NOT DEST)
  message(
    FATAL_ERROR
      "fetch_commit.cmake requires -DGIT_EXECUTABLE, -DREPO, -DTAG and -DDEST")
endif()

file(MAKE_DIRECTORY "${DEST}")

if(NOT EXISTS "${DEST}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" init -q
    WORKING_DIRECTORY "${DEST}"
    RESULT_VARIABLE _r)
  if(_r)
    message(FATAL_ERROR "git init failed in ${DEST}")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" remote add origin "${REPO}"
                  WORKING_DIRECTORY "${DEST}")
endif()

# Keep the remote URL current in case REPO changed between configures.
execute_process(COMMAND "${GIT_EXECUTABLE}" remote set-url origin "${REPO}"
                WORKING_DIRECTORY "${DEST}")

message(STATUS "Shallow-fetching ${TAG} from ${REPO}")
execute_process(
  COMMAND "${GIT_EXECUTABLE}" fetch -q --depth 1 origin "${TAG}"
  WORKING_DIRECTORY "${DEST}"
  RESULT_VARIABLE _rf)
if(_rf)
  message(
    FATAL_ERROR "Shallow fetch of ${TAG} from ${REPO} failed (exit ${_rf}).")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" checkout -q --detach FETCH_HEAD
  WORKING_DIRECTORY "${DEST}"
  RESULT_VARIABLE _rc)
if(_rc)
  message(FATAL_ERROR "git checkout FETCH_HEAD failed in ${DEST} (exit ${_rc}).")
endif()
