include(FetchContent)

if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    )
else()
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
    FIND_PACKAGE_ARGS NAMES GTest # This is a CMake 3.24 feature
    )
endif()

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

function(add_gtest)
    get_filename_component(test_name ${ARGV0} NAME_WE)
    add_executable(${test_name} ${ARGV})
    target_link_libraries(${test_name} PRIVATE frsz GTest::gtest_main GTest::gtest gmock LibPressio::libpressio)
    target_compile_options(${test_name} PRIVATE 
        $<$<CONFIG:Debug>: -Wall  -Wextra  -Wpedantic>
    )
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    gtest_discover_tests(${test_name})
endfunction()

add_gtest(test_frsz.cc)

if (BUILD_CUDA_TESTING)
    set(source_file test_frsz_gpu.cu)
    get_filename_component(test_name ${source_file}  NAME_WE)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gtest gmock)
    get_target_property(frsz_include_directories frsz INCLUDE_DIRECTORIES)
    target_include_directories(${test_name} PRIVATE ${frsz_include_directories})
    target_compile_options(${test_name} PRIVATE 
        $<$<CONFIG:Debug>: -Wall  -Wextra>
    )
    target_compile_options(${test_name} PRIVATE 
        --expt-extended-lambda
        --expt-relaxed-constexpr # TODO remove this and implement missing functions
        )
    target_compile_features(${test_name} PRIVATE cxx_std_14 cuda_std_14)
    # For now, ensure that C++14 is used without extensions.
    # TODO Can be removed after implementation is finalized and properly tested.
    set_target_properties(${test_name} PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        CXX_STANDARD 14
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
        CUDA_STANDARD 14
    )
endif()

# vim: ft=cmake :
