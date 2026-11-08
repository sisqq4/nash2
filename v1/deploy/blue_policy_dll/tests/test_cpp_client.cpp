#include "blue_policy_client.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

blue_policy::StepRequest cruise_step(
    std::int64_t aircraft_id,
    const std::string& aircraft_type,
    std::int64_t step_count) {
    blue_policy::StepRequest request;
    request.session_id = blue_policy::Client::default_session_id(aircraft_id);
    request.aircraft_id = aircraft_id;
    request.aircraft_type = aircraft_type;
    request.step_count = step_count;
    request.scenario_time_s = static_cast<double>(step_count) * 0.005;
    request.ownship.position = {
        108.0 + static_cast<double>(aircraft_id % 100) * 1.0e-5,
        34.0,
        10'000.0,
    };
    request.ownship.velocity = {300.0, 0.0, 0.0};
    request.ownship.actual_load =
        blue_policy::LoadCommandBody{0.0, 1.0, 0.0};
    return request;
}

void reset_aircraft(
    blue_policy::Client& client,
    std::int64_t aircraft_id,
    const std::string& aircraft_type,
    blue_policy::InputMode input_mode) {
    blue_policy::ResetRequest request;
    request.session_id = blue_policy::Client::default_session_id(aircraft_id);
    request.aircraft_id = aircraft_id;
    request.aircraft_type = aircraft_type;
    request.episode_id = 1;
    request.input_mode = input_mode;
    const blue_policy::ResetResult result = client.reset(request);
    require(result.aircraft_id == aircraft_id, "reset returned the wrong aircraft");
    require(result.aircraft_type == aircraft_type, "aircraft type was not preserved");
    require(result.input_mode == input_mode, "reset returned the wrong input mode");
}

void test_single_aircraft_and_string_escaping(blue_policy::Client& client) {
    constexpr std::int64_t kAircraftId = 9'007'199'254'740'993LL;
    const std::string aircraft_type = "fighter-\"quoted\\type";
    reset_aircraft(
        client,
        kAircraftId,
        aircraft_type,
        blue_policy::InputMode::KinematicsOnly);

    blue_policy::StepRequest request =
        cruise_step(kAircraftId, aircraft_type, 0);
    request.include_diagnostics = true;
    const blue_policy::StepResult result = client.step(request);

    require(result.aircraft_id == kAircraftId, "int64 aircraft ID lost precision");
    require(result.step_count == 0, "step returned the wrong step_count");
    require(
        std::abs(result.velocity_effective_time_s - 0.005) < 1.0e-12,
        "step returned the wrong velocity time");
    require(
        std::abs(result.next_velocity.north_mps - 300.0) < 1.0e-9 &&
            std::abs(result.next_velocity.up_mps) < 1.0e-9 &&
            std::abs(result.next_velocity.east_mps) < 1.0e-9,
        "cruise step returned the wrong N/U/E velocity");
    require(result.diagnostics.has_value(), "diagnostics were not parsed");
    require(
        result.raw_response_json.find("\"action_index\"") == std::string::npos &&
            result.raw_response_json.find("\"raw_action_index\"") ==
                std::string::npos,
        "public JSON response exposed an action index");

    const std::string session_id =
        blue_policy::Client::default_session_id(kAircraftId);
    require(
        client.close_session(session_id, kAircraftId).closed,
        "first close did not close the session");
    require(
        !client.close_session(session_id, kAircraftId).closed,
        "second close was not idempotent");
}

void test_business_errors_are_typed_and_atomic(blue_policy::Client& client) {
    constexpr std::int64_t kAircraftId = 102;
    const std::string aircraft_type = "fighter-exact";
    reset_aircraft(
        client,
        kAircraftId,
        aircraft_type,
        blue_policy::InputMode::Exact);

    blue_policy::StepRequest request =
        cruise_step(kAircraftId, aircraft_type, 0);
    blue_policy::IncomingMissile missile;
    missile.missile_id = 9'001;
    missile.position = {108.0, 33.55, 10'000.0};
    missile.velocity = {1'000.0, 0.0, 0.0};
    request.incoming_missiles.push_back(missile);

    bool missing_fields_rejected = false;
    try {
        static_cast<void>(client.step(request));
    } catch (const blue_policy::BusinessError& error) {
        missing_fields_rejected = error.code() == "BPJ_MISSING_FIELD" &&
            error.state_changed().has_value() && !*error.state_changed() &&
            !error.reset_required();
    }
    require(
        missing_fields_rejected,
        "exact-mode missing fields did not produce an atomic business error");

    request.incoming_missiles[0].energy = 0.6;
    request.incoming_missiles[0].guidance_mode =
        blue_policy::GuidanceMode::Locked;
    const blue_policy::StepResult first = client.step(request);
    require(first.active, "visible exact-mode missile did not activate the policy");

    bool sequence_rejected = false;
    try {
        static_cast<void>(client.step(request));
    } catch (const blue_policy::BusinessError& error) {
        sequence_rejected = error.code() == "BPJ_SEQUENCE_ERROR" &&
            error.state_changed().has_value() && !*error.state_changed() &&
            !error.reset_required();
    }
    require(
        sequence_rejected,
        "duplicate frame did not produce BPJ_SEQUENCE_ERROR");

    request.step_count = 1;
    request.scenario_time_s = 0.005;
    const blue_policy::StepResult second = client.step(request);
    require(second.step_count == 1, "session did not recover after sequence error");

    require(
        client.close_session(request.session_id, kAircraftId).closed,
        "exact-mode session was not closed");
}

void test_multi_aircraft_concurrency(blue_policy::Client& client) {
    constexpr int kAircraftCount = 8;
    constexpr int kFrameCount = 3;
    const std::string aircraft_type = "fighter-concurrent";
    for (int index = 0; index < kAircraftCount; ++index) {
        reset_aircraft(
            client,
            200 + index,
            aircraft_type,
            blue_policy::InputMode::KinematicsOnly);
    }

    std::atomic<bool> success{true};
    std::mutex error_mutex;
    std::string first_error;
    std::vector<std::thread> workers;
    workers.reserve(kAircraftCount);
    for (int index = 0; index < kAircraftCount; ++index) {
        workers.emplace_back([&, index]() {
            try {
                const std::int64_t aircraft_id = 200 + index;
                for (int frame = 0; frame < kFrameCount; ++frame) {
                    const blue_policy::StepResult result = client.step(
                        cruise_step(aircraft_id, aircraft_type, frame));
                    require(
                        result.aircraft_id == aircraft_id,
                        "concurrent result was delivered to the wrong aircraft");
                    require(
                        result.step_count == frame,
                        "concurrent result has the wrong step_count");
                    require(
                        std::isfinite(result.next_velocity.north_mps) &&
                            std::isfinite(result.next_velocity.up_mps) &&
                            std::isfinite(result.next_velocity.east_mps),
                        "concurrent result contains a non-finite velocity");
                }
            } catch (const std::exception& error) {
                success.store(false, std::memory_order_release);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                    first_error = error.what();
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    require(
        success.load(std::memory_order_acquire),
        "multi-aircraft concurrency failed: " + first_error);

    for (int index = 0; index < kAircraftCount; ++index) {
        const std::int64_t aircraft_id = 200 + index;
        require(
            client.close_session(
                      blue_policy::Client::default_session_id(aircraft_id),
                      aircraft_id)
                .closed,
            "concurrent session was not closed");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "usage: blue_policy_client_test <python_app_dir> "
               "[python_home]\n";
        return 2;
    }
    try {
        blue_policy::InitializationOptions options;
        options.python_module_dir_utf8 = argv[1];
        options.bundle_dir_utf8 = argv[1];
        if (argc == 3) {
            options.python_home_utf8 = argv[2];
        }

        blue_policy::Client client(options);
        const blue_policy::InfoResult info = client.info();
        require(
            info.dll_api_version == BLUE_POLICY_API_VERSION,
            "info returned the wrong DLL API version");
        require(
            info.json_protocol_version == 1U,
            "info returned the wrong JSON protocol version");
        require(
            info.concurrency_mode == "serialized_gil_v1",
            "info returned an unexpected concurrency mode");
        require(
            std::abs(info.physics_step_s - 0.005) < 1.0e-12,
            "info returned the wrong physics step");

        test_single_aircraft_and_string_escaping(client);
        test_business_errors_are_typed_and_atomic(client);
        test_multi_aircraft_concurrency(client);

        client.shutdown();
        std::cout
            << "typed C++ client integration passed: model=" << info.model_id
            << ", concurrent_aircraft=8\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "typed C++ client integration failed: " << error.what()
                  << '\n';
        return 1;
    }
}
