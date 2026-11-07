#ifndef BLUE_POLICY_API_H
#define BLUE_POLICY_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(BLUE_POLICY_BRIDGE_BUILD)
#    define BLUE_POLICY_API __declspec(dllexport)
#  else
#    define BLUE_POLICY_API __declspec(dllimport)
#  endif
#  define BLUE_POLICY_CALL __cdecl
#else
#  define BLUE_POLICY_API __attribute__((visibility("default")))
#  define BLUE_POLICY_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BLUE_POLICY_API_VERSION 2u
#define BLUE_POLICY_OBSERVATION_DIM 104u
#define BLUE_POLICY_ACTION_COUNT 29u
#define BLUE_POLICY_MAX_VISIBLE_TRACKS 3u
#define BLUE_POLICY_MAX_JSON_REQUEST_BYTES 1048576u

typedef enum BluePolicyStatus {
    BLUE_POLICY_STATUS_OK = 0,
    BLUE_POLICY_STATUS_INVALID_ARGUMENT = 1,
    BLUE_POLICY_STATUS_NOT_INITIALIZED = 2,
    BLUE_POLICY_STATUS_ALREADY_INITIALIZED = 3,
    BLUE_POLICY_STATUS_PYTHON_ERROR = 4,
    BLUE_POLICY_STATUS_WRONG_THREAD = 5,
    BLUE_POLICY_STATUS_INTERNAL_ERROR = 6,
    BLUE_POLICY_STATUS_UNSUPPORTED_STATE = 7
} BluePolicyStatus;

typedef enum BluePolicyInputMode {
    BLUE_POLICY_INPUT_EXACT = 0,
    BLUE_POLICY_INPUT_KINEMATICS_ONLY = 1
} BluePolicyInputMode;

typedef enum BluePolicyGuidanceMode {
    BLUE_POLICY_GUIDANCE_BOOST = 0,
    BLUE_POLICY_GUIDANCE_LOCKED = 1,
    BLUE_POLICY_GUIDANCE_LOCK_HOLD = 2,
    BLUE_POLICY_GUIDANCE_INERTIAL = 3
} BluePolicyGuidanceMode;

typedef enum BluePolicyRuntimeMode {
    BLUE_POLICY_MODE_INACTIVE = 0,
    BLUE_POLICY_MODE_CRUISE = 1,
    BLUE_POLICY_MODE_RAINBOW = 2
} BluePolicyRuntimeMode;

/*
 * All strings use UTF-8. python_home_utf8 may be NULL when CPython has
 * already been initialized by the host or is discoverable by the platform.
 * python_module_dir_utf8 must contain blue_policy_entry.py. bundle_dir_utf8
 * must contain blue_policy_runtime/ and model/.
 */
typedef struct BluePolicyInitConfig {
    uint32_t struct_size;
    const char* python_home_utf8;
    const char* python_module_dir_utf8;
    const char* bundle_dir_utf8;
} BluePolicyInitConfig;

typedef struct BluePolicyTrack {
    int64_t track_id;
    double position_m[3];
    double velocity_mps[3];
    int32_t alive;
    double energy;
    int32_t guidance_mode;
} BluePolicyTrack;

typedef struct BluePolicyFrame {
    uint32_t struct_size;
    int64_t step_count;
    double time_s;
    double blue_position_m[3];
    double blue_velocity_mps[3];
    int32_t blue_alive;
    double actual_load_command_body_g[3];
    const BluePolicyTrack* tracks;
    uint32_t track_count;
} BluePolicyFrame;

typedef struct BluePolicyOutput {
    uint32_t struct_size;
    int32_t mode;
    int32_t active;
    int32_t decision_updated;
    int32_t raw_action_index;
    int32_t action_index;
    double target_command_body_g[3];
    double applied_command_body_g[3];
    double valid_until_s;
    int32_t action_intervened;
    int32_t safety_intervened;
    int32_t fallback_required;
    int32_t safe_action_count;
    int64_t decision_count;
    uint32_t visible_track_count;
    int64_t visible_track_ids[BLUE_POLICY_MAX_VISIBLE_TRACKS];
} BluePolicyOutput;

/*
 * API v2 JSON result. The DLL allocates data_utf8 and appends a trailing NUL;
 * size_bytes excludes that NUL. Initialize the whole structure to zero, then
 * set struct_size before calling blue_policy_call_json. Always release with
 * blue_policy_release_json from the same loaded DLL.
 */
typedef struct BluePolicyJsonResult {
    uint32_t struct_size;
    char* data_utf8;
    size_t size_bytes;
} BluePolicyJsonResult;

BLUE_POLICY_API uint32_t BLUE_POLICY_CALL blue_policy_api_version(void);

BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_initialize(
    const BluePolicyInitConfig* config);

/* Direct 104-float C51 inference. Returns the raw argmax action in [0, 28]. */
BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_predict(
    const float* observation,
    uint32_t observation_length,
    int32_t* out_action);

/* Creates one independent stateful blue-aircraft policy session. */
BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_create_session(
    int32_t input_mode,
    int64_t episode_id,
    uint64_t* out_session);

BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_reset_session(
    uint64_t session,
    int64_t episode_id);

BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_step(
    uint64_t session,
    const BluePolicyFrame* frame,
    BluePolicyOutput* out_output);

/*
 * Preferred API v2 business interface. Concurrent calls from different
 * native threads are safe; this release serializes execution internally.
 * A native return value of OK means a JSON response was produced. Business
 * validation failures are represented by {"ok":false,...} in that response.
 */
BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_call_json(
    const char* request_utf8,
    size_t request_size_bytes,
    BluePolicyJsonResult* out_result);

BLUE_POLICY_API void BLUE_POLICY_CALL blue_policy_release_json(
    BluePolicyJsonResult* result);

BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_destroy_session(
    uint64_t session);

/*
 * The returned pointer is owned by the bridge and stays valid until the next
 * API call on the same native thread. Do not free it.
 */
BLUE_POLICY_API const char* BLUE_POLICY_CALL blue_policy_last_error(void);

/*
 * If this bridge initialized CPython, shutdown is terminal for this process
 * and must run on the same native thread as blue_policy_initialize().
 */
BLUE_POLICY_API int32_t BLUE_POLICY_CALL blue_policy_shutdown(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* BLUE_POLICY_API_H */
