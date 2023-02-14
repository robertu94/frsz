include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    FIND_PACKAGE_ARGS NAMES GTest
    )
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

function(add_gtest)
  get_filename_component(test_name ${ARGV0} NAME_WE)
  add_executable(${test_name} ${ARGV})
  target_link_libraries(${test_name} PRIVATE frsz GTest::gtest_main GTest::gtest gmock LibPressio::libpressio)
  target_compile_options(${test_name} PRIVATE 
      $<$<CONFIG:Debug>: -Wall  -Wextra -Wpedantic>
  )
  gtest_discover_tests(${test_name})
endfunction()

add_gtest(test_frsz.cc)

# vim: ft=cmake :
