cmake_minimum_required(VERSION 3.18)

project(BluePolicyBridge VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BLUE_POLICY_BUILD_TEST_HOST "Build the native C++ tests" ON)
option(
    BLUE_POLICY_BUILD_CLIENT_EXAMPLE
    "Build the reusable typed C++ client example"
    ON
)
set(
    BLUE_POLICY_RUNTIME_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../blue_policy_exe"
    CACHE PATH
    "Directory containing blue_policy_runtime/ and model/"
)

find_package(
    Python3 3.11 EXACT REQUIRED
    COMPONENTS Interpreter Development.Embed
)

foreach(required_path
        "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/blue_policy_runtime"
        "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/model/manifest.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/python_app/blue_policy_entry.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python_app/blue_policy_json.py")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Required blue policy runtime path is missing: ${required_path}")
    endif()
endforeach()

add_library(blue_policy_bridge SHARED
    src/blue_policy_bridge.cpp
)
target_compile_definitions(blue_policy_bridge PRIVATE BLUE_POLICY_BRIDGE_BUILD)
target_include_directories(blue_policy_bridge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(blue_policy_bridge PRIVATE Python3::Python)
set_target_properties(blue_policy_bridge PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(MSVC)
    target_compile_options(blue_policy_bridge PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(blue_policy_bridge PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Reusable C++17 wrapper for the JSON C ABI. This is a static client library so
# STL objects and C++ exceptions never cross a DLL boundary. The final host also
# links the blue_policy_bridge import library through this PUBLIC dependency.
add_library(blue_policy_client STATIC
    src/blue_policy_client.cpp
)
add_library(BluePolicy::client ALIAS blue_policy_client)
target_include_directories(blue_policy_client
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(blue_policy_client PUBLIC blue_policy_bridge)
target_compile_features(blue_policy_client PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(blue_policy_client PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(blue_policy_client PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Stage the Python entry, frozen runtime, and model beside the native binary.
# The bridge still needs a compatible CPython + NumPy installation; this step
# only makes the project-specific portion self-contained.
set(BLUE_POLICY_STAGED_APP_DIR "$<TARGET_FILE_DIR:blue_policy_bridge>/python_app")
add_custom_target(blue_policy_stage ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${BLUE_POLICY_STAGED_APP_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/python_app/blue_policy_entry.py"
        "${BLUE_POLICY_STAGED_APP_DIR}/blue_policy_entry.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/python_app/blue_policy_json.py"
        "${BLUE_POLICY_STAGED_APP_DIR}/blue_policy_json.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/blue_policy_runtime"
        "${BLUE_POLICY_STAGED_APP_DIR}/blue_policy_runtime"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/model"
        "${BLUE_POLICY_STAGED_APP_DIR}/model"
    DEPENDS blue_policy_bridge
    SOURCES python_app/blue_policy_entry.py python_app/blue_policy_json.py
    VERBATIM
)

get_filename_component(_python_executable_dir "${Python3_EXECUTABLE}" DIRECTORY)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    RESULT_VARIABLE _python_prefix_status
    OUTPUT_VARIABLE _python_home
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _python_prefix_status EQUAL 0 OR _python_home STREQUAL "")
    message(FATAL_ERROR "Could not query sys.prefix from ${Python3_EXECUTABLE}")
endif()
if(WIN32)
    set(
        _python_runtime_dll
        "${_python_executable_dir}/python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll"
    )
    set(_python_runtime_files "${_python_runtime_dll}")
    if(EXISTS "${_python_executable_dir}/zlib.dll")
        list(APPEND _python_runtime_files "${_python_executable_dir}/zlib.dll")
    endif()
    foreach(_python_runtime_file IN LISTS _python_runtime_files)
        if(NOT EXISTS "${_python_runtime_file}")
            message(FATAL_ERROR "Python runtime dependency is missing: ${_python_runtime_file}")
        endif()
        add_custom_command(TARGET blue_policy_bridge POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_python_runtime_file}"
                "$<TARGET_FILE_DIR:blue_policy_bridge>"
            VERBATIM
        )
    endforeach()
endif()

if(BLUE_POLICY_BUILD_TEST_HOST)
    enable_testing()
    add_executable(blue_policy_test_host examples/test_host.cpp)
    target_link_libraries(blue_policy_test_host PRIVATE blue_policy_bridge)
    add_dependencies(blue_policy_test_host blue_policy_stage)
    if(APPLE)
        set_target_properties(blue_policy_test_host PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(blue_policy_test_host PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    if(MSVC)
        target_compile_options(blue_policy_test_host PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(blue_policy_test_host PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(
        NAME blue_policy_native_smoke
        COMMAND blue_policy_test_host
            "$<TARGET_FILE_DIR:blue_policy_bridge>/python_app"
            "${_python_home}"
    )

    add_executable(blue_policy_client_test tests/test_cpp_client.cpp)
    target_link_libraries(blue_policy_client_test PRIVATE BluePolicy::client)
    add_dependencies(blue_policy_client_test blue_policy_stage)
    if(MSVC)
        target_compile_options(blue_policy_client_test PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(
            blue_policy_client_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(
        NAME blue_policy_cpp_client_integration
        COMMAND blue_policy_client_test
            "$<TARGET_FILE_DIR:blue_policy_bridge>/python_app"
            "${_python_home}"
    )
endif()

if(BLUE_POLICY_BUILD_CLIENT_EXAMPLE)
    add_executable(
        blue_policy_client_example
        examples/blue_policy_client_example.cpp
    )
    target_link_libraries(blue_policy_client_example PRIVATE BluePolicy::client)
    add_dependencies(blue_policy_client_example blue_policy_stage)
    if(MSVC)
        target_compile_options(
            blue_policy_client_example PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(
            blue_policy_client_example PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

include(GNUInstallDirs)
install(
    TARGETS blue_policy_bridge
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
    TARGETS blue_policy_client
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
if(WIN32)
    install(FILES ${_python_runtime_files} DESTINATION .)
endif()
install(
    FILES include/blue_policy_api.h include/blue_policy_client.hpp
    DESTINATION include
)
install(FILES requirements-inference.txt DESTINATION .)
install(FILES README.md DESTINATION .)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/BLUE_POLICY_DLL_DESIGN.md"
    DESTINATION docs
)
install(
    FILES examples/blue_policy_client_example.cpp
    DESTINATION examples
)
install(
    FILES tests/test_cpp_client.cpp
    DESTINATION tests
)
install(
    FILES python_app/blue_policy_entry.py python_app/blue_policy_json.py
    DESTINATION python_app
)
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/schemas/blue_policy_request_v1.schema.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/schemas/blue_policy_response_v1.schema.json"
    DESTINATION schemas
)
install(
    DIRECTORY "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/blue_policy_runtime"
    DESTINATION python_app
    PATTERN "__pycache__" EXCLUDE
)
install(
    DIRECTORY "${BLUE_POLICY_RUNTIME_SOURCE_DIR}/model"
    DESTINATION python_app
)
if(BLUE_POLICY_BUILD_TEST_HOST)
    install(
        TARGETS blue_policy_test_host blue_policy_client_test
        RUNTIME DESTINATION .
    )
endif()
if(BLUE_POLICY_BUILD_CLIENT_EXAMPLE)
    install(TARGETS blue_policy_client_example RUNTIME DESTINATION .)
endif()
