#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#endif

#include "blue_policy_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr uint32_t kMaxTracksPerFrame = 4096;
constexpr size_t kMaxJsonResponseBytes = 4u * 1024u * 1024u;

std::mutex g_api_mutex;
thread_local std::string g_last_error;

bool g_initialized = false;
bool g_owns_python = false;
bool g_owned_interpreter_was_finalized = false;
std::thread::id g_initialization_thread;
PyThreadState* g_saved_thread_state = nullptr;
std::string g_process_module_directory;
std::string g_process_bundle_directory;

PyObject* g_module = nullptr;
PyObject* g_initialize_function = nullptr;
PyObject* g_dispatch_json_function = nullptr;
PyObject* g_predict_function = nullptr;
PyObject* g_create_session_function = nullptr;
PyObject* g_reset_session_function = nullptr;
PyObject* g_step_function = nullptr;
PyObject* g_destroy_session_function = nullptr;
PyObject* g_shutdown_function = nullptr;

class PyReference {
public:
    explicit PyReference(PyObject* value = nullptr) noexcept : value_(value) {}
    ~PyReference() { Py_XDECREF(value_); }

    PyReference(const PyReference&) = delete;
    PyReference& operator=(const PyReference&) = delete;

    PyReference(PyReference&& other) noexcept : value_(other.value_) {
        other.value_ = nullptr;
    }

    PyReference& operator=(PyReference&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(value_);
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] PyObject* get() const noexcept { return value_; }
    [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }
    [[nodiscard]] PyObject* release() noexcept {
        PyObject* result = value_;
        value_ = nullptr;
        return result;
    }

private:
    PyObject* value_;
};

class GilGuard {
public:
    GilGuard() : state_(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state_); }

    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;

private:
    PyGILState_STATE state_;
};

void clear_last_error() { g_last_error.clear(); }

int32_t fail(BluePolicyStatus status, std::string message) {
    g_last_error = std::move(message);
    return static_cast<int32_t>(status);
}

std::string object_text(PyObject* object) {
    if (object == nullptr) {
        return {};
    }
    PyReference text(PyObject_Str(object));
    if (!text) {
        PyErr_Clear();
        return {};
    }
    const char* utf8 = PyUnicode_AsUTF8(text.get());
    if (utf8 == nullptr) {
        PyErr_Clear();
        return {};
    }
    return utf8;
}

std::string consume_python_error() {
    if (!PyErr_Occurred()) {
        return "unknown Python error";
    }

    PyObject* raw_type = nullptr;
    PyObject* raw_value = nullptr;
    PyObject* raw_traceback = nullptr;
    PyErr_Fetch(&raw_type, &raw_value, &raw_traceback);
    PyErr_NormalizeException(&raw_type, &raw_value, &raw_traceback);
    PyReference type(raw_type);
    PyReference value(raw_value);
    PyReference traceback(raw_traceback);

    PyReference traceback_module(PyImport_ImportModule("traceback"));
    if (traceback_module) {
        PyReference formatter(
            PyObject_GetAttrString(traceback_module.get(), "format_exception"));
        if (formatter && PyCallable_Check(formatter.get())) {
            PyReference formatted(PyObject_CallFunctionObjArgs(
                formatter.get(), type ? type.get() : Py_None,
                value ? value.get() : Py_None,
                traceback ? traceback.get() : Py_None, nullptr));
            PyReference separator(PyUnicode_FromString(""));
            if (formatted && separator) {
                PyReference joined(PyUnicode_Join(separator.get(), formatted.get()));
                if (joined) {
                    const char* utf8 = PyUnicode_AsUTF8(joined.get());
                    if (utf8 != nullptr) {
                        return utf8;
                    }
                }
            }
        }
        PyErr_Clear();
    } else {
        PyErr_Clear();
    }

    std::string type_name;
    if (type) {
        PyReference name(PyObject_GetAttrString(type.get(), "__name__"));
        if (name) {
            const char* utf8 = PyUnicode_AsUTF8(name.get());
            if (utf8 != nullptr) {
                type_name = utf8;
            } else {
                PyErr_Clear();
            }
        } else {
            PyErr_Clear();
        }
    }
    const std::string message = object_text(value.get());
    if (type_name.empty()) {
        return message.empty() ? "unknown Python error" : message;
    }
    return message.empty() ? type_name : type_name + ": " + message;
}

int32_t fail_python(const char* context) {
    return fail(
        BLUE_POLICY_STATUS_PYTHON_ERROR,
        std::string(context) + ": " + consume_python_error());
}

bool is_finite_vector(const double values[3]) {
    return values != nullptr && std::isfinite(values[0]) &&
           std::isfinite(values[1]) && std::isfinite(values[2]);
}

PyReference make_vector(const double values[3]) {
    return PyReference(Py_BuildValue("(ddd)", values[0], values[1], values[2]));
}

bool set_tuple_item(PyObject* tuple, Py_ssize_t index, PyReference value) {
    if (!value) {
        return false;
    }
    PyTuple_SET_ITEM(tuple, index, value.release());
    return true;
}

bool set_list_item(PyObject* list, Py_ssize_t index, PyReference value) {
    if (!value) {
        return false;
    }
    PyList_SET_ITEM(list, index, value.release());
    return true;
}

PyReference make_tracks(const BluePolicyFrame& frame) {
    PyReference result(PyList_New(static_cast<Py_ssize_t>(frame.track_count)));
    if (!result) {
        return PyReference();
    }
    for (uint32_t index = 0; index < frame.track_count; ++index) {
        const BluePolicyTrack& track = frame.tracks[index];
        PyReference item(PyTuple_New(6));
        if (!item ||
            !set_tuple_item(item.get(), 0, PyReference(PyLong_FromLongLong(track.track_id))) ||
            !set_tuple_item(item.get(), 1, make_vector(track.position_m)) ||
            !set_tuple_item(item.get(), 2, make_vector(track.velocity_mps)) ||
            !set_tuple_item(item.get(), 3, PyReference(PyBool_FromLong(track.alive != 0))) ||
            !set_tuple_item(item.get(), 4, PyReference(PyFloat_FromDouble(track.energy))) ||
            !set_tuple_item(
                item.get(), 5, PyReference(PyLong_FromLong(track.guidance_mode))) ||
            !set_list_item(result.get(), static_cast<Py_ssize_t>(index), std::move(item))) {
            return PyReference();
        }
    }
    return result;
}

void clear_python_references() {
    Py_CLEAR(g_shutdown_function);
    Py_CLEAR(g_destroy_session_function);
    Py_CLEAR(g_step_function);
    Py_CLEAR(g_reset_session_function);
    Py_CLEAR(g_create_session_function);
    Py_CLEAR(g_predict_function);
    Py_CLEAR(g_dispatch_json_function);
    Py_CLEAR(g_initialize_function);
    Py_CLEAR(g_module);
}

PyObject* load_callable(PyObject* module, const char* name) {
    PyObject* value = PyObject_GetAttrString(module, name);
    if (value == nullptr) {
        return nullptr;
    }
    if (!PyCallable_Check(value)) {
        Py_DECREF(value);
        PyErr_Format(PyExc_TypeError, "blue_policy_entry.%s is not callable", name);
        return nullptr;
    }
    return value;
}

bool add_module_directory(const char* path_utf8) {
    PyObject* sys_path = PySys_GetObject("path");  // Borrowed reference.
    if (sys_path == nullptr || !PyList_Check(sys_path)) {
        PyErr_SetString(PyExc_RuntimeError, "sys.path is unavailable");
        return false;
    }
    PyReference path(PyUnicode_FromString(path_utf8));
    return path && PyList_Insert(sys_path, 0, path.get()) == 0;
}

bool load_entry_module(
    const char* module_dir_utf8,
    const char* bundle_dir_utf8,
    bool owns_python) {
    if (!add_module_directory(module_dir_utf8)) {
        return false;
    }

    g_module = PyImport_ImportModule("blue_policy_entry");
    if (g_module == nullptr) {
        return false;
    }
    g_initialize_function = load_callable(g_module, "initialize");
    g_dispatch_json_function = load_callable(g_module, "dispatch_json");
    g_predict_function = load_callable(g_module, "predict");
    g_create_session_function = load_callable(g_module, "create_session");
    g_reset_session_function = load_callable(g_module, "reset_session");
    g_step_function = load_callable(g_module, "step");
    g_destroy_session_function = load_callable(g_module, "destroy_session");
    g_shutdown_function = load_callable(g_module, "shutdown");
    if (g_initialize_function == nullptr || g_dispatch_json_function == nullptr ||
        g_predict_function == nullptr ||
        g_create_session_function == nullptr || g_reset_session_function == nullptr ||
        g_step_function == nullptr || g_destroy_session_function == nullptr ||
        g_shutdown_function == nullptr) {
        return false;
    }

    PyReference bundle_dir(PyUnicode_FromString(bundle_dir_utf8));
    if (!bundle_dir) {
        return false;
    }
    PyReference ownership(PyUnicode_FromString(owns_python ? "owned" : "borrowed"));
    if (!ownership) {
        return false;
    }
    PyReference result(PyObject_CallFunctionObjArgs(
        g_initialize_function, bundle_dir.get(), ownership.get(), nullptr));
    return static_cast<bool>(result);
}

std::wstring decode_path(const char* path_utf8) {
    if (path_utf8 == nullptr || *path_utf8 == '\0') {
        return {};
    }
#if defined(_WIN32)
    const int required = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, nullptr, 0);
    if (required <= 0) {
        throw std::runtime_error("python_home_utf8 is not valid UTF-8");
    }
    std::wstring result(static_cast<size_t>(required), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, result.data(),
            required) != required) {
        throw std::runtime_error("could not decode python_home_utf8");
    }
    result.pop_back();
    return result;
#else
    size_t length = 0;
    wchar_t* decoded = Py_DecodeLocale(path_utf8, &length);
    if (decoded == nullptr) {
        throw std::runtime_error("could not decode python_home_utf8");
    }
    std::wstring result(decoded, length);
    PyMem_RawFree(decoded);
    return result;
#endif
}

int32_t initialize_owned_python(const char* python_home_utf8) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    config.write_bytecode = 0;

    PyStatus status = PyStatus_Ok();
    try {
        const std::wstring home = decode_path(python_home_utf8);
        if (!home.empty()) {
            status = PyConfig_SetString(&config, &config.home, home.c_str());
        }
    } catch (const std::exception& error) {
        PyConfig_Clear(&config);
        return fail(BLUE_POLICY_STATUS_INVALID_ARGUMENT, error.what());
    }
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    const std::string status_error =
        status.err_msg == nullptr ? "CPython initialization failed" : status.err_msg;
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        return fail(BLUE_POLICY_STATUS_PYTHON_ERROR, status_error);
    }
    return BLUE_POLICY_STATUS_OK;
}

bool parse_long(PyObject* value, long long& output) {
    if (value == nullptr) {
        PyErr_SetString(PyExc_TypeError, "missing integer result");
        return false;
    }
    output = PyLong_AsLongLong(value);
    return !(output == -1 && PyErr_Occurred());
}

bool parse_double(PyObject* value, double& output) {
    output = PyFloat_AsDouble(value);
    if (output == -1.0 && PyErr_Occurred()) {
        return false;
    }
    if (!std::isfinite(output)) {
        PyErr_SetString(PyExc_ValueError, "step result contains a non-finite value");
        return false;
    }
    return true;
}

bool parse_boolean(PyObject* value, int32_t& output) {
    const int truth = PyObject_IsTrue(value);
    if (truth < 0) {
        return false;
    }
    output = truth != 0 ? 1 : 0;
    return true;
}

bool parse_vector(PyObject* value, double output[3]) {
    PyReference sequence(PySequence_Fast(value, "expected a three-element sequence"));
    if (!sequence || PySequence_Fast_GET_SIZE(sequence.get()) != 3) {
        if (sequence) {
            PyErr_SetString(PyExc_ValueError, "expected a three-element sequence");
        }
        return false;
    }
    PyObject** items = PySequence_Fast_ITEMS(sequence.get());
    return parse_double(items[0], output[0]) && parse_double(items[1], output[1]) &&
           parse_double(items[2], output[2]);
}

bool parse_step_result(PyObject* value, BluePolicyOutput& output) {
    if (!PyTuple_Check(value) || PyTuple_GET_SIZE(value) != 14) {
        PyErr_SetString(
            PyExc_TypeError, "blue_policy_entry.step must return a 14-element tuple");
        return false;
    }

    long long integer = 0;
    if (!parse_long(PyTuple_GET_ITEM(value, 0), integer) || integer < 0 || integer > 2) {
        PyErr_SetString(PyExc_ValueError, "invalid runtime mode in step result");
        return false;
    }
    output.mode = static_cast<int32_t>(integer);
    if (!parse_boolean(PyTuple_GET_ITEM(value, 1), output.active) ||
        !parse_boolean(PyTuple_GET_ITEM(value, 2), output.decision_updated)) {
        return false;
    }
    if (!parse_long(PyTuple_GET_ITEM(value, 3), integer) || integer < 0 ||
        integer >= BLUE_POLICY_ACTION_COUNT) {
        PyErr_SetString(PyExc_ValueError, "invalid raw action in step result");
        return false;
    }
    output.raw_action_index = static_cast<int32_t>(integer);
    if (!parse_long(PyTuple_GET_ITEM(value, 4), integer) || integer < 0 ||
        integer >= BLUE_POLICY_ACTION_COUNT) {
        PyErr_SetString(PyExc_ValueError, "invalid action in step result");
        return false;
    }
    output.action_index = static_cast<int32_t>(integer);
    if (!parse_vector(PyTuple_GET_ITEM(value, 5), output.target_command_body_g) ||
        !parse_vector(PyTuple_GET_ITEM(value, 6), output.applied_command_body_g) ||
        !parse_double(PyTuple_GET_ITEM(value, 7), output.valid_until_s) ||
        !parse_boolean(PyTuple_GET_ITEM(value, 8), output.action_intervened) ||
        !parse_boolean(PyTuple_GET_ITEM(value, 9), output.safety_intervened) ||
        !parse_boolean(PyTuple_GET_ITEM(value, 10), output.fallback_required)) {
        return false;
    }
    if (!parse_long(PyTuple_GET_ITEM(value, 11), integer) || integer < 0 ||
        integer > BLUE_POLICY_ACTION_COUNT) {
        PyErr_SetString(PyExc_ValueError, "invalid safe action count in step result");
        return false;
    }
    output.safe_action_count = static_cast<int32_t>(integer);
    if (!parse_long(PyTuple_GET_ITEM(value, 12), integer) || integer < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid decision count in step result");
        return false;
    }
    output.decision_count = static_cast<int64_t>(integer);

    PyReference visible(PySequence_Fast(
        PyTuple_GET_ITEM(value, 13), "visible track IDs must be a sequence"));
    if (!visible) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(visible.get());
    if (count < 0 || count > BLUE_POLICY_MAX_VISIBLE_TRACKS) {
        PyErr_SetString(PyExc_ValueError, "too many visible track IDs in step result");
        return false;
    }
    output.visible_track_count = static_cast<uint32_t>(count);
    PyObject** items = PySequence_Fast_ITEMS(visible.get());
    for (Py_ssize_t index = 0; index < count; ++index) {
        if (!parse_long(items[index], integer)) {
            return false;
        }
        output.visible_track_ids[index] = static_cast<int64_t>(integer);
    }
    return true;
}

int32_t require_initialized() {
    if (!g_initialized) {
        return fail(
            BLUE_POLICY_STATUS_NOT_INITIALIZED,
            "blue policy is not initialized; call blue_policy_initialize first");
    }
    return BLUE_POLICY_STATUS_OK;
}

}  // namespace

extern "C" {

uint32_t BLUE_POLICY_CALL blue_policy_api_version(void) {
    return BLUE_POLICY_API_VERSION;
}

int32_t BLUE_POLICY_CALL blue_policy_initialize(const BluePolicyInitConfig* config) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    bool created_interpreter = false;
    bool released_owned_gil = false;
    const auto rollback = [&]() noexcept {
        if (created_interpreter) {
            if (released_owned_gil && g_saved_thread_state != nullptr) {
                PyEval_RestoreThread(g_saved_thread_state);
                g_saved_thread_state = nullptr;
            }
            if (Py_IsInitialized()) {
                clear_python_references();
                Py_FinalizeEx();
            }
            g_owns_python = false;
            g_owned_interpreter_was_finalized = true;
        } else if (Py_IsInitialized()) {
            GilGuard gil;
            clear_python_references();
        }
        g_initialized = false;
    };
    try {
        if (config == nullptr || config->struct_size < sizeof(BluePolicyInitConfig)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "config is null or has an incompatible struct_size");
        }
        if (config->python_module_dir_utf8 == nullptr ||
            *config->python_module_dir_utf8 == '\0' ||
            config->bundle_dir_utf8 == nullptr || *config->bundle_dir_utf8 == '\0') {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "python_module_dir_utf8 and bundle_dir_utf8 are required");
        }
        if (g_initialized) {
            return fail(
                BLUE_POLICY_STATUS_ALREADY_INITIALIZED,
                "blue policy is already initialized");
        }
        if (g_owned_interpreter_was_finalized) {
            return fail(
                BLUE_POLICY_STATUS_UNSUPPORTED_STATE,
                "an interpreter owned by this bridge was already finalized; "
                "restart the process before initializing again");
        }
        if ((!g_process_module_directory.empty() &&
             g_process_module_directory != config->python_module_dir_utf8) ||
            (!g_process_bundle_directory.empty() &&
             g_process_bundle_directory != config->bundle_dir_utf8)) {
            return fail(
                BLUE_POLICY_STATUS_UNSUPPORTED_STATE,
                "the Python module and bundle directories cannot change within one process");
        }

        const bool borrowed_interpreter = Py_IsInitialized() != 0;
        if (!borrowed_interpreter) {
            const int32_t status = initialize_owned_python(config->python_home_utf8);
            if (status != BLUE_POLICY_STATUS_OK) {
                return status;
            }
            g_owns_python = true;
            created_interpreter = true;
        }

        bool loaded = false;
        if (borrowed_interpreter) {
            GilGuard gil;
            loaded = load_entry_module(
                config->python_module_dir_utf8, config->bundle_dir_utf8, false);
            if (!loaded) {
                const int32_t status = fail_python("failed to initialize Python entry module");
                clear_python_references();
                return status;
            }
        } else {
            loaded = load_entry_module(
                config->python_module_dir_utf8, config->bundle_dir_utf8, true);
            if (!loaded) {
                const int32_t status = fail_python("failed to initialize Python entry module");
                clear_python_references();
                Py_FinalizeEx();
                g_owns_python = false;
                g_owned_interpreter_was_finalized = true;
                created_interpreter = false;
                return status;
            }
            g_saved_thread_state = PyEval_SaveThread();
            released_owned_gil = true;
        }

        g_initialized = true;
        g_initialization_thread = std::this_thread::get_id();
        g_process_module_directory = config->python_module_dir_utf8;
        g_process_bundle_directory = config->bundle_dir_utf8;
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        const std::string message = error.what();
        rollback();
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, message);
    } catch (...) {
        rollback();
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

int32_t BLUE_POLICY_CALL blue_policy_predict(
    const float* observation,
    uint32_t observation_length,
    int32_t* out_action) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (observation == nullptr || out_action == nullptr ||
            observation_length != BLUE_POLICY_OBSERVATION_DIM) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "predict requires a non-null 104-float observation and output pointer");
        }
        for (uint32_t index = 0; index < observation_length; ++index) {
            if (!std::isfinite(observation[index])) {
                return fail(
                    BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                    "observation contains a non-finite value");
            }
        }

        GilGuard gil;
        PyReference values(PyList_New(observation_length));
        if (!values) {
            return fail_python("could not allocate observation list");
        }
        for (uint32_t index = 0; index < observation_length; ++index) {
            if (!set_list_item(
                    values.get(), static_cast<Py_ssize_t>(index),
                    PyReference(PyFloat_FromDouble(observation[index])))) {
                return fail_python("could not convert observation");
            }
        }
        PyReference result(PyObject_CallOneArg(g_predict_function, values.get()));
        if (!result) {
            return fail_python("blue policy prediction failed");
        }
        long long action = 0;
        if (!parse_long(result.get(), action)) {
            return fail_python("blue policy returned a non-integer action");
        }
        if (action < 0 || action >= BLUE_POLICY_ACTION_COUNT) {
            return fail(
                BLUE_POLICY_STATUS_PYTHON_ERROR,
                "blue policy returned an action outside [0, 28]");
        }
        *out_action = static_cast<int32_t>(action);
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

int32_t BLUE_POLICY_CALL blue_policy_create_session(
    int32_t input_mode,
    int64_t episode_id,
    uint64_t* out_session) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (out_session == nullptr ||
            (input_mode != BLUE_POLICY_INPUT_EXACT &&
             input_mode != BLUE_POLICY_INPUT_KINEMATICS_ONLY)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "create_session requires a valid input mode and output pointer");
        }

        GilGuard gil;
        PyReference mode(PyLong_FromLong(input_mode));
        PyReference episode(PyLong_FromLongLong(episode_id));
        if (!mode || !episode) {
            return fail_python("could not convert session arguments");
        }
        PyReference result(PyObject_CallFunctionObjArgs(
            g_create_session_function, mode.get(), episode.get(), nullptr));
        if (!result) {
            return fail_python("could not create blue policy session");
        }
        const unsigned long long handle = PyLong_AsUnsignedLongLong(result.get());
        if (handle == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
            return fail_python("blue policy returned an invalid session handle");
        }
        if (handle == 0) {
            return fail(
                BLUE_POLICY_STATUS_PYTHON_ERROR,
                "blue policy returned the reserved zero session handle");
        }
        *out_session = static_cast<uint64_t>(handle);
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

int32_t BLUE_POLICY_CALL blue_policy_reset_session(
    uint64_t session,
    int64_t episode_id) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (session == 0) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT, "session handle must not be zero");
        }
        GilGuard gil;
        PyReference handle(PyLong_FromUnsignedLongLong(session));
        PyReference episode(PyLong_FromLongLong(episode_id));
        if (!handle || !episode) {
            return fail_python("could not convert reset arguments");
        }
        PyReference result(PyObject_CallFunctionObjArgs(
            g_reset_session_function, handle.get(), episode.get(), nullptr));
        if (!result) {
            return fail_python("could not reset blue policy session");
        }
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

int32_t BLUE_POLICY_CALL blue_policy_step(
    uint64_t session,
    const BluePolicyFrame* frame,
    BluePolicyOutput* out_output) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (session == 0 || frame == nullptr || out_output == nullptr ||
            frame->struct_size < sizeof(BluePolicyFrame) ||
            out_output->struct_size < sizeof(BluePolicyOutput)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "step received a null pointer, zero session, or incompatible struct_size");
        }
        if (frame->step_count < 0 || !std::isfinite(frame->time_s) ||
            frame->time_s < 0.0 || !is_finite_vector(frame->blue_position_m) ||
            !is_finite_vector(frame->blue_velocity_mps) ||
            !is_finite_vector(frame->actual_load_command_body_g) ||
            (frame->blue_alive != 0 && frame->blue_alive != 1)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "frame contains an invalid time, vector, step count, or alive flag");
        }
        if (frame->track_count > kMaxTracksPerFrame ||
            (frame->track_count != 0 && frame->tracks == nullptr)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "track_count exceeds 4096 or tracks is null");
        }
        for (uint32_t index = 0; index < frame->track_count; ++index) {
            const BluePolicyTrack& track = frame->tracks[index];
            if (!is_finite_vector(track.position_m) ||
                !is_finite_vector(track.velocity_mps) ||
                !std::isfinite(track.energy) ||
                (track.alive != 0 && track.alive != 1) || track.guidance_mode < 0 ||
                track.guidance_mode > 3) {
                std::ostringstream message;
                message << "track " << index << " contains an invalid field";
                return fail(BLUE_POLICY_STATUS_INVALID_ARGUMENT, message.str());
            }
        }

        GilGuard gil;
        PyReference handle(PyLong_FromUnsignedLongLong(session));
        PyReference step_count(PyLong_FromLongLong(frame->step_count));
        PyReference time(PyFloat_FromDouble(frame->time_s));
        PyReference position(make_vector(frame->blue_position_m));
        PyReference velocity(make_vector(frame->blue_velocity_mps));
        PyReference alive(PyBool_FromLong(frame->blue_alive));
        PyReference actual(make_vector(frame->actual_load_command_body_g));
        PyReference tracks(make_tracks(*frame));
        if (!handle || !step_count || !time || !position || !velocity || !alive ||
            !actual || !tracks) {
            return fail_python("could not convert step arguments");
        }

        PyReference result(PyObject_CallFunctionObjArgs(
            g_step_function, handle.get(), step_count.get(), time.get(), position.get(),
            velocity.get(), alive.get(), actual.get(), tracks.get(), nullptr));
        if (!result) {
            return fail_python("blue policy step failed");
        }
        BluePolicyOutput parsed{};
        parsed.struct_size = sizeof(BluePolicyOutput);
        if (!parse_step_result(result.get(), parsed)) {
            return fail_python("could not parse blue policy step result");
        }
        *out_output = parsed;
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

int32_t BLUE_POLICY_CALL blue_policy_call_json(
    const char* request_utf8,
    size_t request_size_bytes,
    BluePolicyJsonResult* out_result) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        if (out_result == nullptr ||
            out_result->struct_size < sizeof(BluePolicyJsonResult)) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "JSON result is null or has an incompatible struct_size");
        }
        if (out_result->data_utf8 != nullptr || out_result->size_bytes != 0) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "JSON result must be empty before blue_policy_call_json");
        }
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (request_utf8 == nullptr || request_size_bytes == 0 ||
            request_size_bytes > BLUE_POLICY_MAX_JSON_REQUEST_BYTES) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "JSON request must contain 1 to 1048576 bytes");
        }
        if (std::memchr(request_utf8, '\0', request_size_bytes) != nullptr) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT,
                "JSON request contains an embedded NUL byte");
        }

        GilGuard gil;
        PyReference request(PyUnicode_DecodeUTF8(
            request_utf8, static_cast<Py_ssize_t>(request_size_bytes), "strict"));
        if (!request) {
            return fail_python("JSON request is not valid UTF-8");
        }
        PyReference response(
            PyObject_CallOneArg(g_dispatch_json_function, request.get()));
        if (!response) {
            return fail_python("blue policy JSON dispatch failed");
        }
        if (!PyUnicode_Check(response.get())) {
            return fail(
                BLUE_POLICY_STATUS_PYTHON_ERROR,
                "blue_policy_entry.dispatch_json must return a string");
        }
        Py_ssize_t response_size = 0;
        const char* response_utf8 =
            PyUnicode_AsUTF8AndSize(response.get(), &response_size);
        if (response_utf8 == nullptr || response_size < 0) {
            return fail_python("could not encode blue policy JSON response");
        }
        if (static_cast<size_t>(response_size) > kMaxJsonResponseBytes) {
            return fail(
                BLUE_POLICY_STATUS_PYTHON_ERROR,
                "blue policy JSON response exceeds 4 MiB");
        }
        char* buffer = new (std::nothrow) char[static_cast<size_t>(response_size) + 1];
        if (buffer == nullptr) {
            return fail(
                BLUE_POLICY_STATUS_INTERNAL_ERROR,
                "could not allocate blue policy JSON response");
        }
        std::memcpy(buffer, response_utf8, static_cast<size_t>(response_size));
        buffer[response_size] = '\0';
        out_result->data_utf8 = buffer;
        out_result->size_bytes = static_cast<size_t>(response_size);
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

void BLUE_POLICY_CALL blue_policy_release_json(BluePolicyJsonResult* result) {
    if (result == nullptr || result->struct_size < sizeof(BluePolicyJsonResult)) {
        return;
    }
    delete[] result->data_utf8;
    result->data_utf8 = nullptr;
    result->size_bytes = 0;
}

int32_t BLUE_POLICY_CALL blue_policy_destroy_session(uint64_t session) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        const int32_t ready = require_initialized();
        if (ready != BLUE_POLICY_STATUS_OK) {
            return ready;
        }
        if (session == 0) {
            return fail(
                BLUE_POLICY_STATUS_INVALID_ARGUMENT, "session handle must not be zero");
        }
        GilGuard gil;
        PyReference handle(PyLong_FromUnsignedLongLong(session));
        if (!handle) {
            return fail_python("could not convert session handle");
        }
        PyReference result(
            PyObject_CallOneArg(g_destroy_session_function, handle.get()));
        if (!result) {
            return fail_python("could not destroy blue policy session");
        }
        return BLUE_POLICY_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

const char* BLUE_POLICY_CALL blue_policy_last_error(void) {
    return g_last_error.c_str();
}

int32_t BLUE_POLICY_CALL blue_policy_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    clear_last_error();
    try {
        if (!g_initialized) {
            return BLUE_POLICY_STATUS_OK;
        }
        if (g_owns_python && std::this_thread::get_id() != g_initialization_thread) {
            return fail(
                BLUE_POLICY_STATUS_WRONG_THREAD,
                "shutdown must run on the same native thread as initialization");
        }

        int32_t result_status = BLUE_POLICY_STATUS_OK;
        if (g_owns_python) {
            PyEval_RestoreThread(g_saved_thread_state);
            g_saved_thread_state = nullptr;
            {
                PyReference result(PyObject_CallNoArgs(g_shutdown_function));
                if (!result) {
                    result_status = fail_python("Python policy shutdown failed");
                }
            }
            clear_python_references();
            if (Py_FinalizeEx() < 0 && result_status == BLUE_POLICY_STATUS_OK) {
                result_status = fail(
                    BLUE_POLICY_STATUS_PYTHON_ERROR, "Py_FinalizeEx reported failure");
            }
            g_owned_interpreter_was_finalized = true;
        } else {
            GilGuard gil;
            PyReference result(PyObject_CallNoArgs(g_shutdown_function));
            if (!result) {
                result_status = fail_python("Python policy shutdown failed");
            }
            clear_python_references();
        }

        g_initialized = false;
        g_owns_python = false;
        g_initialization_thread = std::thread::id{};
        return result_status;
    } catch (const std::exception& error) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(BLUE_POLICY_STATUS_INTERNAL_ERROR, "unknown native exception");
    }
}

}  // extern "C"
