#include "blue_policy_client.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <utility>

namespace blue_policy {
namespace {

struct JsonValue {
    enum class Type {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object,
    };

    Type type{Type::Null};
    bool boolean{};
    std::string text;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser final {
public:
    explicit JsonParser(std::string_view source) : source_(source) {}

    [[nodiscard]] JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value(0);
        skip_whitespace();
        if (position_ != source_.size()) {
            fail("unexpected trailing data");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(
            "invalid JSON response at byte " + std::to_string(position_) +
            ": " + message);
    }

    void skip_whitespace() {
        while (position_ < source_.size()) {
            const char value = source_[position_];
            if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
                break;
            }
            ++position_;
        }
    }

    [[nodiscard]] bool consume(char expected) {
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void require(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    [[nodiscard]] JsonValue parse_value(std::size_t depth) {
        if (depth > 64) {
            fail("nesting depth exceeds 64");
        }
        skip_whitespace();
        if (position_ >= source_.size()) {
            fail("unexpected end of input");
        }
        switch (source_[position_]) {
        case 'n':
            return parse_literal("null", JsonValue::Type::Null, false);
        case 't':
            return parse_literal("true", JsonValue::Type::Boolean, true);
        case 'f':
            return parse_literal("false", JsonValue::Type::Boolean, false);
        case '"': {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.text = parse_string();
            return value;
        }
        case '[':
            return parse_array(depth + 1);
        case '{':
            return parse_object(depth + 1);
        default:
            if (source_[position_] == '-' ||
                (source_[position_] >= '0' && source_[position_] <= '9')) {
                return parse_number();
            }
            fail("unexpected token");
        }
    }

    [[nodiscard]] JsonValue parse_literal(
        std::string_view literal,
        JsonValue::Type type,
        bool boolean) {
        if (source_.substr(position_, literal.size()) != literal) {
            fail("invalid literal");
        }
        position_ += literal.size();
        JsonValue value;
        value.type = type;
        value.boolean = boolean;
        return value;
    }

    [[nodiscard]] static int hex_value(char value) {
        if (value >= '0' && value <= '9') {
            return value - '0';
        }
        if (value >= 'a' && value <= 'f') {
            return value - 'a' + 10;
        }
        if (value >= 'A' && value <= 'F') {
            return value - 'A' + 10;
        }
        return -1;
    }

    [[nodiscard]] std::uint32_t parse_hex4() {
        if (source_.size() - position_ < 4) {
            fail("incomplete Unicode escape");
        }
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const int digit = hex_value(source_[position_++]);
            if (digit < 0) {
                fail("invalid Unicode escape");
            }
            value = value * 16U + static_cast<std::uint32_t>(digit);
        }
        return value;
    }

    static void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7FU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0x10FFFFU) {
            output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else {
            throw std::runtime_error("JSON Unicode codepoint is out of range");
        }
    }

    [[nodiscard]] std::string parse_string() {
        require('"');
        std::string output;
        while (position_ < source_.size()) {
            const unsigned char value =
                static_cast<unsigned char>(source_[position_++]);
            if (value == static_cast<unsigned char>('"')) {
                return output;
            }
            if (value < 0x20U) {
                fail("unescaped control character in string");
            }
            if (value != static_cast<unsigned char>('\\')) {
                output.push_back(static_cast<char>(value));
                continue;
            }
            if (position_ >= source_.size()) {
                fail("incomplete string escape");
            }
            const char escape = source_[position_++];
            switch (escape) {
            case '"':
            case '\\':
            case '/':
                output.push_back(escape);
                break;
            case 'b':
                output.push_back('\b');
                break;
            case 'f':
                output.push_back('\f');
                break;
            case 'n':
                output.push_back('\n');
                break;
            case 'r':
                output.push_back('\r');
                break;
            case 't':
                output.push_back('\t');
                break;
            case 'u': {
                std::uint32_t codepoint = parse_hex4();
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                    if (source_.size() - position_ < 6 ||
                        source_[position_] != '\\' ||
                        source_[position_ + 1] != 'u') {
                        fail("high surrogate is not followed by a low surrogate");
                    }
                    position_ += 2;
                    const std::uint32_t low = parse_hex4();
                    if (low < 0xDC00U || low > 0xDFFFU) {
                        fail("invalid low surrogate");
                    }
                    codepoint = 0x10000U +
                        ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
                } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                    fail("unexpected low surrogate");
                }
                append_utf8(output, codepoint);
                break;
            }
            default:
                fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }

    [[nodiscard]] JsonValue parse_number() {
        const std::size_t start = position_;
        static_cast<void>(consume('-'));
        if (position_ >= source_.size()) {
            fail("incomplete number");
        }
        if (source_[position_] == '0') {
            ++position_;
            if (position_ < source_.size() && source_[position_] >= '0' &&
                source_[position_] <= '9') {
                fail("leading zero in number");
            }
        } else {
            if (source_[position_] < '1' || source_[position_] > '9') {
                fail("invalid number");
            }
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
        }
        if (consume('.')) {
            if (position_ >= source_.size() || source_[position_] < '0' ||
                source_[position_] > '9') {
                fail("fraction has no digits");
            }
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < source_.size() &&
            (source_[position_] == 'e' || source_[position_] == 'E')) {
            ++position_;
            if (position_ < source_.size() &&
                (source_[position_] == '+' || source_[position_] == '-')) {
                ++position_;
            }
            if (position_ >= source_.size() || source_[position_] < '0' ||
                source_[position_] > '9') {
                fail("exponent has no digits");
            }
            while (position_ < source_.size() && source_[position_] >= '0' &&
                   source_[position_] <= '9') {
                ++position_;
            }
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.text = std::string(source_.substr(start, position_ - start));
        return value;
    }

    [[nodiscard]] JsonValue parse_array(std::size_t depth) {
        require('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        skip_whitespace();
        if (consume(']')) {
            return value;
        }
        while (true) {
            value.array.push_back(parse_value(depth));
            skip_whitespace();
            if (consume(']')) {
                return value;
            }
            require(',');
            skip_whitespace();
        }
    }

    [[nodiscard]] JsonValue parse_object(std::size_t depth) {
        require('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        skip_whitespace();
        if (consume('}')) {
            return value;
        }
        while (true) {
            if (position_ >= source_.size() || source_[position_] != '"') {
                fail("object key must be a string");
            }
            std::string key = parse_string();
            skip_whitespace();
            require(':');
            JsonValue member = parse_value(depth);
            if (!value.object.emplace(std::move(key), std::move(member)).second) {
                fail("duplicate object key");
            }
            skip_whitespace();
            if (consume('}')) {
                return value;
            }
            require(',');
            skip_whitespace();
        }
    }

    std::string_view source_;
    std::size_t position_{};
};

[[nodiscard]] const JsonValue& require_member(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::Object) {
        throw std::runtime_error("JSON response value is not an object");
    }
    const auto iterator = value.object.find(std::string(name));
    if (iterator == value.object.end()) {
        throw std::runtime_error(
            "JSON response is missing field " + std::string(name));
    }
    return iterator->second;
}

[[nodiscard]] const JsonValue* find_member(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::Object) {
        throw std::runtime_error("JSON response value is not an object");
    }
    const auto iterator = value.object.find(std::string(name));
    return iterator == value.object.end() ? nullptr : &iterator->second;
}

[[nodiscard]] bool as_bool(const JsonValue& value, std::string_view name) {
    if (value.type != JsonValue::Type::Boolean) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not boolean");
    }
    return value.boolean;
}

[[nodiscard]] const std::string& as_string(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::String) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not a string");
    }
    return value.text;
}

[[nodiscard]] std::int64_t as_int64(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::Number) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not a number");
    }
    std::int64_t result = 0;
    const char* first = value.text.data();
    const char* last = first + value.text.size();
    const auto parsed = std::from_chars(first, last, result);
    if (parsed.ec != std::errc{} || parsed.ptr != last) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not an int64");
    }
    return result;
}

[[nodiscard]] std::uint32_t as_uint32(
    const JsonValue& value,
    std::string_view name) {
    const std::int64_t result = as_int64(value, name);
    if (result < 0 ||
        result > static_cast<std::int64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) +
            " is outside uint32 range");
    }
    return static_cast<std::uint32_t>(result);
}

[[nodiscard]] double as_double(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::Number) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not a number");
    }
    std::istringstream stream(value.text);
    stream.imbue(std::locale::classic());
    double result = 0.0;
    stream >> result;
    if (!stream || !stream.eof() || !std::isfinite(result)) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) +
            " is not a finite double");
    }
    return result;
}

[[nodiscard]] const std::vector<JsonValue>& as_array(
    const JsonValue& value,
    std::string_view name) {
    if (value.type != JsonValue::Type::Array) {
        throw std::runtime_error(
            "JSON response field " + std::string(name) + " is not an array");
    }
    return value.array;
}

[[nodiscard]] VelocityNue parse_velocity(const JsonValue& value) {
    const auto& items = as_array(value, "next_velocity_north_up_east_mps");
    if (items.size() != 3) {
        throw std::runtime_error(
            "next_velocity_north_up_east_mps must have three elements");
    }
    return {
        as_double(items[0], "next_velocity[0]"),
        as_double(items[1], "next_velocity[1]"),
        as_double(items[2], "next_velocity[2]"),
    };
}

[[nodiscard]] LoadCommandBody parse_load(const JsonValue& value) {
    const auto& items = as_array(value, "applied_load_command_body_g");
    if (items.size() != 3) {
        throw std::runtime_error(
            "applied_load_command_body_g must have three elements");
    }
    return {
        as_double(items[0], "applied_load[0]"),
        as_double(items[1], "applied_load[1]"),
        as_double(items[2], "applied_load[2]"),
    };
}

[[nodiscard]] InputMode parse_input_mode(const JsonValue& value) {
    const std::string& mode = as_string(value, "input_mode");
    if (mode == "exact") {
        return InputMode::Exact;
    }
    if (mode == "kinematics_only") {
        return InputMode::KinematicsOnly;
    }
    throw std::runtime_error("JSON response contains an unsupported input_mode");
}

[[nodiscard]] const JsonValue& require_success_result(
    const JsonValue& envelope,
    const std::string& raw_response,
    std::uint64_t expected_request_id,
    std::string_view expected_operation) {
    const std::int64_t protocol_version = as_int64(
        require_member(envelope, "protocol_version"), "protocol_version");
    if (protocol_version != 1) {
        throw std::runtime_error(
            "blue policy response has an unexpected protocol_version");
    }
    const std::int64_t echoed_request_id =
        as_int64(require_member(envelope, "request_id"), "request_id");
    if (echoed_request_id < 0 ||
        static_cast<std::uint64_t>(echoed_request_id) != expected_request_id) {
        throw std::runtime_error(
            "blue policy response request_id does not match the request");
    }
    const bool ok = as_bool(require_member(envelope, "ok"), "ok");
    if (ok) {
        const JsonValue& result = require_member(envelope, "result");
        if (as_string(require_member(result, "operation"), "operation") !=
            expected_operation) {
            throw std::runtime_error(
                "blue policy response operation does not match the request");
        }
        return result;
    }

    const JsonValue& error = require_member(envelope, "error");
    const std::string code = as_string(require_member(error, "code"), "code");
    const std::string message =
        as_string(require_member(error, "message"), "message");
    const JsonValue& raw_path = require_member(error, "path");
    const std::string path = raw_path.type == JsonValue::Type::Null
        ? std::string()
        : as_string(raw_path, "path");
    const bool retryable =
        as_bool(require_member(error, "retryable"), "retryable");
    const JsonValue& raw_state_changed = require_member(error, "state_changed");
    std::optional<bool> state_changed;
    if (raw_state_changed.type != JsonValue::Type::Null) {
        state_changed = as_bool(raw_state_changed, "state_changed");
    }
    const bool reset_required =
        as_bool(require_member(error, "reset_required"), "reset_required");
    throw BusinessError(
        code,
        message,
        path,
        retryable,
        state_changed,
        reset_required,
        raw_response);
}

[[nodiscard]] std::string quote_json(std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (byte < 0x20U) {
                result += "\\u00";
                result.push_back(kHex[(byte >> 4U) & 0x0FU]);
                result.push_back(kHex[byte & 0x0FU]);
            } else {
                result.push_back(static_cast<char>(byte));
            }
            break;
        }
    }
    result.push_back('"');
    return result;
}

[[nodiscard]] std::string format_number(double value, std::string_view name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(name) + " must be a finite number");
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return stream.str();
}

[[nodiscard]] std::string input_mode_text(InputMode mode) {
    switch (mode) {
    case InputMode::Exact:
        return "exact";
    case InputMode::KinematicsOnly:
        return "kinematics_only";
    }
    throw std::invalid_argument("unsupported input mode");
}

[[nodiscard]] std::string guidance_mode_text(GuidanceMode mode) {
    switch (mode) {
    case GuidanceMode::Boost:
        return "boost";
    case GuidanceMode::Locked:
        return "locked";
    case GuidanceMode::LockHold:
        return "lock_hold";
    case GuidanceMode::Inertial:
        return "inertial";
    }
    throw std::invalid_argument("unsupported guidance mode");
}

[[nodiscard]] std::string serialize_position(const GeodeticPosition& value) {
    return std::string("{\"longitude_deg\":") +
        format_number(value.longitude_deg, "longitude_deg") +
        ",\"latitude_deg\":" +
        format_number(value.latitude_deg, "latitude_deg") +
        ",\"altitude_m\":" +
        format_number(value.altitude_m, "altitude_m") + '}';
}

[[nodiscard]] std::string serialize_velocity(const VelocityNue& value) {
    return std::string("[") +
        format_number(value.north_mps, "velocity.north_mps") + ',' +
        format_number(value.up_mps, "velocity.up_mps") + ',' +
        format_number(value.east_mps, "velocity.east_mps") + ']';
}

[[nodiscard]] std::string serialize_load(const LoadCommandBody& value) {
    return std::string("[") +
        format_number(value.axial_g, "load.axial_g") + ',' +
        format_number(value.normal_g, "load.normal_g") + ',' +
        format_number(value.bank_rad, "load.bank_rad") + ']';
}

[[nodiscard]] std::string serialize_ownship(const OwnshipState& value) {
    std::string result = "{\"geodetic_position\":";
    result += serialize_position(value.position);
    result += ",\"velocity_north_up_east_mps\":";
    result += serialize_velocity(value.velocity);
    result += ",\"alive\":";
    result += value.alive ? "true" : "false";
    if (value.actual_load.has_value()) {
        result += ",\"actual_load_command_body_g\":";
        result += serialize_load(*value.actual_load);
    }
    result.push_back('}');
    return result;
}

[[nodiscard]] std::string serialize_missile(const IncomingMissile& value) {
    std::string result = "{\"missile_id\":" +
        std::to_string(value.missile_id);
    if (value.missile_type.has_value()) {
        result += ",\"missile_type\":" + quote_json(*value.missile_type);
    }
    result += ",\"geodetic_position\":";
    result += serialize_position(value.position);
    result += ",\"velocity_north_up_east_mps\":";
    result += serialize_velocity(value.velocity);
    result += ",\"alive\":";
    result += value.alive ? "true" : "false";
    if (value.energy.has_value()) {
        result += ",\"energy\":" +
            format_number(*value.energy, "missile.energy");
    }
    if (value.guidance_mode.has_value()) {
        result += ",\"guidance_mode\":" +
            quote_json(guidance_mode_text(*value.guidance_mode));
    }
    result.push_back('}');
    return result;
}

[[nodiscard]] std::string request_prefix(
    std::uint64_t request_id,
    std::string_view operation) {
    if (request_id >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("blue policy request_id exceeded int64 range");
    }
    return std::string("{\"protocol_version\":1,\"request_id\":") +
        std::to_string(request_id) + ",\"op\":" + quote_json(operation);
}

class JsonResultGuard final {
public:
    JsonResultGuard() { value.struct_size = sizeof(value); }
    ~JsonResultGuard() { blue_policy_release_json(&value); }

    JsonResultGuard(const JsonResultGuard&) = delete;
    JsonResultGuard& operator=(const JsonResultGuard&) = delete;

    BluePolicyJsonResult value{};
};

[[nodiscard]] std::string current_native_error() {
    const char* raw = blue_policy_last_error();
    return raw == nullptr || *raw == '\0' ? "unknown native error" : raw;
}

[[noreturn]] void throw_native(std::int32_t status, std::string_view operation) {
    throw NativeError(
        status,
        std::string(operation) + " failed: " + current_native_error());
}

}  // namespace

NativeError::NativeError(std::int32_t status, const std::string& message)
    : std::runtime_error(message), status_(status) {}

std::int32_t NativeError::status() const noexcept { return status_; }

BusinessError::BusinessError(
    std::string code,
    std::string message,
    std::string path,
    bool retryable,
    std::optional<bool> state_changed,
    bool reset_required,
    std::string raw_response_json)
    : std::runtime_error(
          code + ": " + message + (path.empty() ? "" : " at " + path)),
      code_(std::move(code)),
      path_(std::move(path)),
      retryable_(retryable),
      state_changed_(state_changed),
      reset_required_(reset_required),
      raw_response_json_(std::move(raw_response_json)) {}

const std::string& BusinessError::code() const noexcept { return code_; }
const std::string& BusinessError::path() const noexcept { return path_; }
bool BusinessError::retryable() const noexcept { return retryable_; }
const std::optional<bool>& BusinessError::state_changed() const noexcept {
    return state_changed_;
}
bool BusinessError::reset_required() const noexcept { return reset_required_; }
const std::string& BusinessError::raw_response_json() const noexcept {
    return raw_response_json_;
}

Client::Client(const InitializationOptions& options)
    : initialization_thread_(std::this_thread::get_id()) {
    if (options.python_module_dir_utf8.empty() ||
        options.bundle_dir_utf8.empty()) {
        throw std::invalid_argument(
            "python_module_dir_utf8 and bundle_dir_utf8 are required");
    }
    if (blue_policy_api_version() != BLUE_POLICY_API_VERSION) {
        throw std::runtime_error(
            "blue policy DLL API version is incompatible with the C++ client");
    }

    BluePolicyInitConfig config{};
    config.struct_size = sizeof(config);
    config.python_home_utf8 = options.python_home_utf8.empty()
        ? nullptr
        : options.python_home_utf8.c_str();
    config.python_module_dir_utf8 = options.python_module_dir_utf8.c_str();
    config.bundle_dir_utf8 = options.bundle_dir_utf8.c_str();
    const std::int32_t status = blue_policy_initialize(&config);
    if (status != BLUE_POLICY_STATUS_OK) {
        throw_native(status, "blue_policy_initialize");
    }
    initialized_.store(true, std::memory_order_release);
}

Client::~Client() noexcept {
    if (initialized_.load(std::memory_order_acquire) &&
        std::this_thread::get_id() == initialization_thread_) {
        blue_policy_shutdown();
        initialized_.store(false, std::memory_order_release);
    }
}

std::string Client::call_json_raw(std::string_view request_json) const {
    if (!initialized_.load(std::memory_order_acquire)) {
        throw std::logic_error("blue policy client is not initialized");
    }
    JsonResultGuard result;
    const std::int32_t status = blue_policy_call_json(
        request_json.data(), request_json.size(), &result.value);
    if (status != BLUE_POLICY_STATUS_OK) {
        throw_native(status, "blue_policy_call_json");
    }
    if (result.value.data_utf8 == nullptr || result.value.size_bytes == 0) {
        throw NativeError(
            BLUE_POLICY_STATUS_INTERNAL_ERROR,
            "blue_policy_call_json returned an empty response");
    }
    return std::string(result.value.data_utf8, result.value.size_bytes);
}

std::uint64_t Client::next_request_id() {
    const std::uint64_t value =
        next_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (value >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("blue policy request ID exhausted");
    }
    return value;
}

InfoResult Client::info() {
    const std::uint64_t request_id = next_request_id();
    std::string request = request_prefix(request_id, "info");
    request.push_back('}');
    const std::string raw = call_json_raw(request);
    const JsonValue envelope = JsonParser(raw).parse();
    const JsonValue& result =
        require_success_result(envelope, raw, request_id, "info");

    InfoResult parsed;
    parsed.dll_api_version =
        as_uint32(require_member(result, "dll_api_version"), "dll_api_version");
    parsed.json_protocol_version = as_uint32(
        require_member(result, "json_protocol_version"),
        "json_protocol_version");
    for (const JsonValue& capability :
         as_array(require_member(result, "capabilities"), "capabilities")) {
        parsed.capabilities.push_back(as_string(capability, "capability"));
    }
    parsed.model_id = as_string(require_member(result, "model_id"), "model_id");
    parsed.model_sha256 =
        as_string(require_member(result, "model_sha256"), "model_sha256");
    parsed.python_bundle_sha256 = as_string(
        require_member(result, "python_bundle_sha256"),
        "python_bundle_sha256");
    parsed.python_version = as_string(
        require_member(result, "python_version"), "python_version");
    parsed.numpy_version =
        as_string(require_member(result, "numpy_version"), "numpy_version");
    parsed.interpreter_ownership = as_string(
        require_member(result, "interpreter_ownership"),
        "interpreter_ownership");
    parsed.concurrency_mode = as_string(
        require_member(result, "concurrency_mode"), "concurrency_mode");
    parsed.physics_step_s = as_double(
        require_member(result, "physics_step_s"), "physics_step_s");
    parsed.raw_response_json = raw;
    if (parsed.dll_api_version != BLUE_POLICY_API_VERSION ||
        parsed.json_protocol_version != 1U) {
        throw std::runtime_error(
            "blue policy info response is incompatible with this C++ client");
    }
    return parsed;
}

ResetResult Client::reset(const ResetRequest& request_data) {
    const std::uint64_t request_id = next_request_id();
    std::string request = request_prefix(request_id, "reset");
    request += ",\"session_id\":" + quote_json(request_data.session_id);
    request += ",\"aircraft_id\":" +
        std::to_string(request_data.aircraft_id);
    request += ",\"aircraft_type\":" +
        quote_json(request_data.aircraft_type);
    request += ",\"episode_id\":" +
        std::to_string(request_data.episode_id);
    request += ",\"input_mode\":" +
        quote_json(input_mode_text(request_data.input_mode));
    request.push_back('}');

    const std::string raw = call_json_raw(request);
    const JsonValue envelope = JsonParser(raw).parse();
    const JsonValue& result =
        require_success_result(envelope, raw, request_id, "reset");
    ResetResult parsed;
    parsed.session_id =
        as_string(require_member(result, "session_id"), "session_id");
    parsed.aircraft_id =
        as_int64(require_member(result, "aircraft_id"), "aircraft_id");
    parsed.aircraft_type =
        as_string(require_member(result, "aircraft_type"), "aircraft_type");
    parsed.episode_id =
        as_int64(require_member(result, "episode_id"), "episode_id");
    parsed.input_mode =
        parse_input_mode(require_member(result, "input_mode"));
    if (parsed.session_id != request_data.session_id ||
        parsed.aircraft_id != request_data.aircraft_id ||
        parsed.aircraft_type != request_data.aircraft_type ||
        parsed.episode_id != request_data.episode_id ||
        parsed.input_mode != request_data.input_mode) {
        throw std::runtime_error(
            "blue policy reset response does not match the request");
    }
    return parsed;
}

StepResult Client::step(const StepRequest& request_data) {
    const std::uint64_t request_id = next_request_id();
    std::string request = request_prefix(request_id, "step");
    request += ",\"session_id\":" + quote_json(request_data.session_id);
    request += ",\"aircraft_id\":" +
        std::to_string(request_data.aircraft_id);
    request += ",\"aircraft_type\":" +
        quote_json(request_data.aircraft_type);
    request += ",\"step_count\":" +
        std::to_string(request_data.step_count);
    request += ",\"scenario_time_s\":" +
        format_number(request_data.scenario_time_s, "scenario_time_s");
    request += ",\"ownship\":" + serialize_ownship(request_data.ownship);
    request += ",\"incoming_missiles\":[";
    for (std::size_t index = 0;
         index < request_data.incoming_missiles.size();
         ++index) {
        if (index != 0) {
            request.push_back(',');
        }
        request += serialize_missile(request_data.incoming_missiles[index]);
    }
    request += "],\"include_diagnostics\":";
    request += request_data.include_diagnostics ? "true" : "false";
    request.push_back('}');

    const std::string raw = call_json_raw(request);
    const JsonValue envelope = JsonParser(raw).parse();
    const JsonValue& result =
        require_success_result(envelope, raw, request_id, "step");

    StepResult parsed;
    parsed.session_id =
        as_string(require_member(result, "session_id"), "session_id");
    parsed.aircraft_id =
        as_int64(require_member(result, "aircraft_id"), "aircraft_id");
    parsed.step_count =
        as_int64(require_member(result, "step_count"), "step_count");
    parsed.scenario_time_s = as_double(
        require_member(result, "scenario_time_s"), "scenario_time_s");
    parsed.velocity_effective_time_s = as_double(
        require_member(result, "velocity_effective_time_s"),
        "velocity_effective_time_s");
    parsed.next_velocity = parse_velocity(
        require_member(result, "next_velocity_north_up_east_mps"));
    parsed.active = as_bool(require_member(result, "active"), "active");
    parsed.fallback_required = as_bool(
        require_member(result, "fallback_required"), "fallback_required");
    if (const JsonValue* diagnostics = find_member(result, "diagnostics")) {
        StepDiagnostics item;
        item.mode =
            as_string(require_member(*diagnostics, "mode"), "diagnostics.mode");
        item.decision_updated = as_bool(
            require_member(*diagnostics, "decision_updated"),
            "diagnostics.decision_updated");
        for (const JsonValue& missile_id : as_array(
                 require_member(*diagnostics, "visible_missile_ids"),
                 "diagnostics.visible_missile_ids")) {
            item.visible_missile_ids.push_back(
                as_int64(missile_id, "visible_missile_id"));
        }
        item.decision_count = as_int64(
            require_member(*diagnostics, "decision_count"),
            "diagnostics.decision_count");
        item.action_intervened = as_bool(
            require_member(*diagnostics, "action_intervened"),
            "diagnostics.action_intervened");
        item.safety_intervened = as_bool(
            require_member(*diagnostics, "safety_intervened"),
            "diagnostics.safety_intervened");
        item.applied_load = parse_load(require_member(
            *diagnostics, "applied_load_command_body_g"));
        parsed.diagnostics = std::move(item);
    }
    parsed.raw_response_json = raw;
    if (parsed.session_id != request_data.session_id ||
        parsed.aircraft_id != request_data.aircraft_id ||
        parsed.step_count != request_data.step_count ||
        std::abs(parsed.scenario_time_s - request_data.scenario_time_s) >
            1.0e-9 ||
        std::abs(
            parsed.velocity_effective_time_s -
            (request_data.scenario_time_s + 0.005)) > 1.0e-9) {
        throw std::runtime_error(
            "blue policy step response does not match the request contract");
    }
    return parsed;
}

CloseResult Client::close_session(
    std::string_view session_id,
    std::int64_t aircraft_id) {
    const std::uint64_t request_id = next_request_id();
    std::string request = request_prefix(request_id, "close_session");
    request += ",\"session_id\":" + quote_json(session_id);
    request += ",\"aircraft_id\":" + std::to_string(aircraft_id);
    request.push_back('}');

    const std::string raw = call_json_raw(request);
    const JsonValue envelope = JsonParser(raw).parse();
    const JsonValue& result =
        require_success_result(envelope, raw, request_id, "close_session");
    CloseResult parsed;
    parsed.session_id =
        as_string(require_member(result, "session_id"), "session_id");
    parsed.aircraft_id =
        as_int64(require_member(result, "aircraft_id"), "aircraft_id");
    parsed.closed = as_bool(require_member(result, "closed"), "closed");
    if (parsed.session_id != session_id || parsed.aircraft_id != aircraft_id) {
        throw std::runtime_error(
            "blue policy close response does not match the request");
    }
    return parsed;
}

void Client::shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    if (std::this_thread::get_id() != initialization_thread_) {
        throw NativeError(
            BLUE_POLICY_STATUS_WRONG_THREAD,
            "blue policy shutdown must run on the initialization thread");
    }
    const std::int32_t status = blue_policy_shutdown();
    initialized_.store(false, std::memory_order_release);
    if (status != BLUE_POLICY_STATUS_OK) {
        throw_native(status, "blue_policy_shutdown");
    }
}

bool Client::initialized() const noexcept {
    return initialized_.load(std::memory_order_acquire);
}

std::string Client::default_session_id(std::int64_t aircraft_id) {
    if (aircraft_id < 0) {
        throw std::invalid_argument("aircraft_id must be non-negative");
    }
    return "blue-" + std::to_string(aircraft_id);
}

}  // namespace blue_policy
