#include "blue_policy_api.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

bool check_status(int32_t status, const char* operation) {
    if (status == BLUE_POLICY_STATUS_OK) {
        return true;
    }
    std::cerr << operation << " failed (status " << status
              << "): " << blue_policy_last_error() << '\n';
    return false;
}

bool call_json(const std::string& request, std::string& response) {
    BluePolicyJsonResult result{};
    result.struct_size = sizeof(result);
    const int32_t status = blue_policy_call_json(
        request.data(), request.size(), &result);
    if (!check_status(status, "call_json")) {
        return false;
    }
    response.assign(result.data_utf8, result.size_bytes);
    blue_policy_release_json(&result);
    blue_policy_release_json(&result);  // Sequential duplicate release is safe.
    return true;
}

std::string reset_json(int aircraft_id) {
    std::ostringstream stream;
    stream << R"({"protocol_version":1,"request_id":"reset-)" << aircraft_id
           << R"(","op":"reset","session_id":"blue-)" << aircraft_id
           << R"(","aircraft_id":)" << aircraft_id
           << R"(,"aircraft_type":"fighter-test","episode_id":1,"input_mode":"kinematics_only"})";
    return stream.str();
}

std::string step_json(int aircraft_id, int step_count) {
    const double time_s = static_cast<double>(step_count) * 0.005;
    std::ostringstream stream;
    stream.precision(15);
    stream << R"({"protocol_version":1,"request_id":"step-)" << aircraft_id
           << '-' << step_count << R"(","op":"step","session_id":"blue-)"
           << aircraft_id << R"(","aircraft_id":)" << aircraft_id
           << R"(,"aircraft_type":"fighter-test","step_count":)" << step_count
           << R"(,"scenario_time_s":)" << time_s
           << R"(,"ownship":{"geodetic_position":{"longitude_deg":108.0,"latitude_deg":34.0,"altitude_m":10000.0},)"
           << R"("velocity_north_up_east_mps":[300.0,0.0,0.0],"alive":true},"incoming_missiles":[]})";
    return stream.str();
}

std::string close_json(int aircraft_id) {
    std::ostringstream stream;
    stream << R"({"protocol_version":1,"request_id":"close-)" << aircraft_id
           << R"(","op":"close_session","session_id":"blue-)" << aircraft_id
           << R"(","aircraft_id":)" << aircraft_id << '}';
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: blue_policy_test_host <python_app_dir> [python_home]\n";
        return 2;
    }

    BluePolicyInitConfig config{};
    config.struct_size = sizeof(config);
    config.python_module_dir_utf8 = argv[1];
    config.bundle_dir_utf8 = argv[1];
    config.python_home_utf8 = argc == 3 ? argv[2] : nullptr;
    if (!check_status(blue_policy_initialize(&config), "initialize")) {
        return 1;
    }
    if (blue_policy_api_version() != 2u) {
        std::cerr << "JSON build did not report API version 2\n";
        blue_policy_shutdown();
        return 1;
    }

    std::string json_response;
    if (!call_json(
            R"({"protocol_version":1,"request_id":"info","op":"info"})",
            json_response) ||
        json_response.find(R"("ok":true)") == std::string::npos ||
        json_response.find(R"("dll_api_version":2)") == std::string::npos) {
        std::cerr << "JSON info call failed: " << json_response << '\n';
        blue_policy_shutdown();
        return 1;
    }

    std::array<float, BLUE_POLICY_OBSERVATION_DIM> observation{};
    int32_t direct_action = -1;
    if (!check_status(
            blue_policy_predict(
                observation.data(), static_cast<uint32_t>(observation.size()),
                &direct_action),
            "predict")) {
        blue_policy_shutdown();
        return 1;
    }
    if (blue_policy_predict(observation.data(), 103, &direct_action) !=
            BLUE_POLICY_STATUS_INVALID_ARGUMENT ||
        std::string(blue_policy_last_error()).empty()) {
        std::cerr << "invalid observation length did not produce a C ABI error\n";
        blue_policy_shutdown();
        return 1;
    }

    uint64_t session = 0;
    if (!check_status(
            blue_policy_create_session(BLUE_POLICY_INPUT_EXACT, 7, &session),
            "create_session")) {
        blue_policy_shutdown();
        return 1;
    }

    BluePolicyTrack track{};
    track.track_id = 1001;
    track.position_m[0] = -50'000.0;
    track.position_m[1] = 10'000.0;
    track.velocity_mps[0] = 1'000.0;
    track.alive = 1;
    track.energy = 0.6;
    track.guidance_mode = BLUE_POLICY_GUIDANCE_LOCKED;

    BluePolicyFrame frame{};
    frame.struct_size = sizeof(frame);
    frame.step_count = 0;
    frame.time_s = 0.0;
    frame.blue_position_m[1] = 10'000.0;
    frame.blue_velocity_mps[0] = 300.0;
    frame.blue_alive = 1;
    frame.actual_load_command_body_g[1] = 1.0;
    frame.tracks = &track;
    frame.track_count = 1;

    BluePolicyOutput output{};
    output.struct_size = sizeof(output);
    if (!check_status(blue_policy_step(session, &frame, &output), "step")) {
        blue_policy_destroy_session(session);
        blue_policy_shutdown();
        return 1;
    }
    if (output.mode != BLUE_POLICY_MODE_RAINBOW || output.active == 0 ||
        output.decision_updated == 0 || output.visible_track_count != 1 ||
        output.visible_track_ids[0] != track.track_id || output.action_index < 0 ||
        output.action_index >= static_cast<int32_t>(BLUE_POLICY_ACTION_COUNT)) {
        std::cerr << "step returned an invalid policy result\n";
        blue_policy_destroy_session(session);
        blue_policy_shutdown();
        return 1;
    }
    output.struct_size = sizeof(output);
    if (blue_policy_step(session, &frame, &output) !=
            BLUE_POLICY_STATUS_PYTHON_ERROR ||
        std::string(blue_policy_last_error()).find("step_count") == std::string::npos) {
        std::cerr << "out-of-order frame did not preserve the Python error\n";
        blue_policy_destroy_session(session);
        blue_policy_shutdown();
        return 1;
    }

    if (!check_status(blue_policy_reset_session(session, 8), "reset_session") ||
        !check_status(blue_policy_destroy_session(session), "destroy_session")) {
        blue_policy_shutdown();
        return 1;
    }

    constexpr int kAircraftCount = 8;
    for (int aircraft = 0; aircraft < kAircraftCount; ++aircraft) {
        if (!call_json(reset_json(100 + aircraft), json_response) ||
            json_response.find(R"("ok":true)") == std::string::npos) {
            std::cerr << "JSON reset failed: " << json_response << '\n';
            blue_policy_shutdown();
            return 1;
        }
    }

    std::atomic<bool> concurrent_ok{true};
    std::vector<std::thread> workers;
    workers.reserve(kAircraftCount);
    for (int aircraft = 0; aircraft < kAircraftCount; ++aircraft) {
        workers.emplace_back([aircraft, &concurrent_ok]() {
            const int aircraft_id = 100 + aircraft;
            for (int step = 0; step < 2; ++step) {
                std::string response;
                if (!call_json(step_json(aircraft_id, step), response) ||
                    response.find(R"("ok":true)") == std::string::npos ||
                    response.find(
                        R"("aircraft_id":)" + std::to_string(aircraft_id)) ==
                        std::string::npos ||
                    response.find("next_velocity_north_up_east_mps") ==
                        std::string::npos ||
                    response.find("action_index") != std::string::npos ||
                    response.find("raw_action_index") != std::string::npos) {
                    concurrent_ok.store(false);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (!concurrent_ok.load()) {
        std::cerr << "concurrent multi-aircraft JSON calls failed\n";
        blue_policy_shutdown();
        return 1;
    }
    for (int aircraft = 0; aircraft < kAircraftCount; ++aircraft) {
        if (!call_json(close_json(100 + aircraft), json_response) ||
            json_response.find(R"("closed":true)") == std::string::npos) {
            std::cerr << "JSON close failed: " << json_response << '\n';
            blue_policy_shutdown();
            return 1;
        }
    }

    std::cout << "API version: " << blue_policy_api_version() << '\n'
              << "Direct action: " << direct_action << '\n'
              << "JSON multi-aircraft threads: " << kAircraftCount << '\n';

    if (!check_status(blue_policy_shutdown(), "shutdown")) {
        return 1;
    }
    return 0;
}
