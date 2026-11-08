#ifndef BLUE_POLICY_CLIENT_HPP
#define BLUE_POLICY_CLIENT_HPP

#include "blue_policy_api.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace blue_policy {

enum class InputMode {
    Exact,
    KinematicsOnly,
};

enum class GuidanceMode {
    Boost,
    Locked,
    LockHold,
    Inertial,
};

struct GeodeticPosition {
    double longitude_deg{};
    double latitude_deg{};
    double altitude_m{};
};

struct VelocityNue {
    double north_mps{};
    double up_mps{};
    double east_mps{};
};

struct LoadCommandBody {
    double axial_g{};
    double normal_g{1.0};
    double bank_rad{};
};

struct OwnshipState {
    GeodeticPosition position;
    VelocityNue velocity;
    bool alive{true};
    std::optional<LoadCommandBody> actual_load;
};

struct IncomingMissile {
    std::int64_t missile_id{};
    std::optional<std::string> missile_type;
    GeodeticPosition position;
    VelocityNue velocity;
    bool alive{true};
    std::optional<double> energy;
    std::optional<GuidanceMode> guidance_mode;
};

struct InitializationOptions {
    std::string python_home_utf8;
    std::string python_module_dir_utf8;
    std::string bundle_dir_utf8;
};

struct InfoResult {
    std::uint32_t dll_api_version{};
    std::uint32_t json_protocol_version{};
    std::vector<std::string> capabilities;
    std::string model_id;
    std::string model_sha256;
    std::string python_bundle_sha256;
    std::string python_version;
    std::string numpy_version;
    std::string interpreter_ownership;
    std::string concurrency_mode;
    double physics_step_s{};
    std::string raw_response_json;
};

struct ResetRequest {
    std::string session_id;
    std::int64_t aircraft_id{};
    std::string aircraft_type;
    std::int64_t episode_id{};
    InputMode input_mode{InputMode::KinematicsOnly};
};

struct ResetResult {
    std::string session_id;
    std::int64_t aircraft_id{};
    std::string aircraft_type;
    std::int64_t episode_id{};
    InputMode input_mode{InputMode::KinematicsOnly};
};

struct StepRequest {
    std::string session_id;
    std::int64_t aircraft_id{};
    std::string aircraft_type;
    std::int64_t step_count{};
    double scenario_time_s{};
    OwnshipState ownship;
    std::vector<IncomingMissile> incoming_missiles;
    bool include_diagnostics{false};
};

struct StepDiagnostics {
    std::string mode;
    bool decision_updated{};
    std::vector<std::int64_t> visible_missile_ids;
    std::int64_t decision_count{};
    bool action_intervened{};
    bool safety_intervened{};
    LoadCommandBody applied_load;
};

struct StepResult {
    std::string session_id;
    std::int64_t aircraft_id{};
    std::int64_t step_count{};
    double scenario_time_s{};
    double velocity_effective_time_s{};
    VelocityNue next_velocity;
    bool active{};
    bool fallback_required{};
    std::optional<StepDiagnostics> diagnostics;
    std::string raw_response_json;
};

struct CloseResult {
    std::string session_id;
    std::int64_t aircraft_id{};
    bool closed{};
};

class NativeError final : public std::runtime_error {
public:
    NativeError(std::int32_t status, const std::string& message);
    [[nodiscard]] std::int32_t status() const noexcept;

private:
    std::int32_t status_;
};

class BusinessError final : public std::runtime_error {
public:
    BusinessError(
        std::string code,
        std::string message,
        std::string path,
        bool retryable,
        std::optional<bool> state_changed,
        bool reset_required,
        std::string raw_response_json);

    [[nodiscard]] const std::string& code() const noexcept;
    [[nodiscard]] const std::string& path() const noexcept;
    [[nodiscard]] bool retryable() const noexcept;
    [[nodiscard]] const std::optional<bool>& state_changed() const noexcept;
    [[nodiscard]] bool reset_required() const noexcept;
    [[nodiscard]] const std::string& raw_response_json() const noexcept;

private:
    std::string code_;
    std::string path_;
    bool retryable_{};
    std::optional<bool> state_changed_;
    bool reset_required_{};
    std::string raw_response_json_;
};

/*
 * One client owns the process-wide bridge initialization. info/reset/step/
 * close_session may be called from different native threads. The caller must
 * stop and join all workers before shutdown, and must serialize calls for the
 * same aircraft/session in FIFO order.
 */
class Client final {
public:
    explicit Client(const InitializationOptions& options);
    ~Client() noexcept;

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

    [[nodiscard]] InfoResult info();
    [[nodiscard]] ResetResult reset(const ResetRequest& request);
    [[nodiscard]] StepResult step(const StepRequest& request);
    [[nodiscard]] CloseResult close_session(
        std::string_view session_id,
        std::int64_t aircraft_id);

    /* The raw request must already contain protocol_version/request_id/op. */
    [[nodiscard]] std::string call_json_raw(std::string_view request_json) const;

    /* Must be called after all worker calls finish, on the initialization thread. */
    void shutdown();

    [[nodiscard]] bool initialized() const noexcept;

    [[nodiscard]] static std::string default_session_id(
        std::int64_t aircraft_id);

private:
    [[nodiscard]] std::uint64_t next_request_id();

    std::atomic<std::uint64_t> next_request_id_{1};
    std::atomic<bool> initialized_{false};
    std::thread::id initialization_thread_;
};

}  // namespace blue_policy

#endif  // BLUE_POLICY_CLIENT_HPP
