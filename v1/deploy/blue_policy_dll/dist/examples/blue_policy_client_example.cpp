#include "blue_policy_client.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "usage: blue_policy_client_example <python_app_dir> "
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
        if (info.dll_api_version != BLUE_POLICY_API_VERSION ||
            info.json_protocol_version != 1U) {
            throw std::runtime_error("unexpected blue policy API/protocol version");
        }

        constexpr std::int64_t kAircraftId = 101;
        const std::string aircraft_type = "fighter-a";
        const std::string session_id =
            blue_policy::Client::default_session_id(kAircraftId);

        blue_policy::ResetRequest reset;
        reset.session_id = session_id;
        reset.aircraft_id = kAircraftId;
        reset.aircraft_type = aircraft_type;
        reset.episode_id = 1;
        reset.input_mode = blue_policy::InputMode::KinematicsOnly;
        static_cast<void>(client.reset(reset));

        blue_policy::StepRequest step;
        step.session_id = session_id;
        step.aircraft_id = kAircraftId;
        step.aircraft_type = aircraft_type;
        step.step_count = 0;
        step.scenario_time_s = 0.0;
        step.ownship.position = {108.0, 34.0, 10'000.0};
        step.ownship.velocity = {300.0, 0.0, 0.0};
        step.ownship.alive = true;
        step.ownship.actual_load = blue_policy::LoadCommandBody{0.0, 1.0, 0.0};
        step.include_diagnostics = true;

        blue_policy::IncomingMissile missile;
        missile.missile_id = 9'001;
        missile.missile_type = "surface-to-air";
        missile.position = {108.0, 33.55, 10'000.0};
        missile.velocity = {1'000.0, 0.0, 0.0};
        missile.alive = true;
        step.incoming_missiles.push_back(missile);

        const blue_policy::StepResult result = client.step(step);
        std::cout
            << "model_id=" << info.model_id << '\n'
            << "aircraft_id=" << result.aircraft_id << '\n'
            << "velocity_effective_time_s="
            << result.velocity_effective_time_s << '\n'
            << "next_velocity_north_up_east_mps=["
            << result.next_velocity.north_mps << ','
            << result.next_velocity.up_mps << ','
            << result.next_velocity.east_mps << "]\n"
            << "active=" << (result.active ? "true" : "false") << '\n'
            << "fallback_required="
            << (result.fallback_required ? "true" : "false") << '\n';

        if (result.diagnostics.has_value()) {
            std::cout
                << "mode=" << result.diagnostics->mode << '\n'
                << "visible_missile_count="
                << result.diagnostics->visible_missile_ids.size() << '\n';
        }

        // The large project should apply result.next_velocity at the next
        // 0.005 s frame. Do not also integrate the diagnostic load command.

        const blue_policy::CloseResult closed =
            client.close_session(session_id, kAircraftId);
        if (!closed.closed) {
            throw std::runtime_error("blue policy session was not closed");
        }

        // Stop and join every worker thread before this call. When the DLL
        // owns CPython, shutdown must run on the initialization thread.
        client.shutdown();
        return 0;
    } catch (const blue_policy::BusinessError& error) {
        std::cerr
            << "business error: code=" << error.code()
            << ", path=" << error.path()
            << ", reset_required="
            << (error.reset_required() ? "true" : "false")
            << ", message=" << error.what() << '\n';
    } catch (const blue_policy::NativeError& error) {
        std::cerr
            << "native error: status=" << error.status()
            << ", message=" << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "client error: " << error.what() << '\n';
    }
    return 1;
}
