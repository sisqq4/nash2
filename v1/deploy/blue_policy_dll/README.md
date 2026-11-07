# v1 蓝方推理动态链接库 API v2

本目录把蓝方冻结 Rainbow 策略按以下路径交付给 Windows x64 C++ 大项目：

```text
C++ 主程序
  → 稳定 C ABI：blue_policy_call_json
  → CPython 3.11 嵌入桥接
  → 严格 JSON 协议与多机 session
  → 小型物理状态转换为内部 104 维观测
  → 冻结 NumPy Rainbow 推理、飞行包线和单步动力学
  → 返回本机下一物理帧的北/上/东三轴速度
```

动态库只提供推理，不包含训练、经验回放、优化器、反向传播、绘图或图像输出。

API v2 的首选业务接口是 `blue_policy_call_json`。它接收 UTF-8 JSON，并返回本机编号和
下一物理帧三轴速度；公共 JSON 响应不返回 `action_index` 或 `raw_action_index`。旧 POD
接口继续保留，仅用于已有调用方兼容、golden test 和底层诊断。

完整设计见[蓝方推理动态库设计规范](../../docs/BLUE_POLICY_DLL_DESIGN.md)，机器可读合同见：

- [JSON 请求 Schema](../../docs/schemas/blue_policy_request_v1.schema.json)
- [JSON 响应 Schema](../../docs/schemas/blue_policy_response_v1.schema.json)

## 1. 当前接口能力

公共头文件为 [`include/blue_policy_api.h`](include/blue_policy_api.h)，当前
`BLUE_POLICY_API_VERSION=2`。所有函数使用 `extern "C"` 和 Windows `__cdecl`，C ABI
边界不传递 STL、Python 对象或 C++ 异常。

| 接口 | 用途 |
|---|---|
| `blue_policy_api_version` | 返回原生 API 版本，JSON 客户端要求值为 `2` |
| `blue_policy_initialize` | 初始化或借用 CPython，加载 Python入口、模型和 JSON dispatcher |
| `blue_policy_call_json` | 首选业务接口；处理 `info/reset/step/close_session` |
| `blue_policy_release_json` | 释放同一 DLL 分配的 JSON 响应内存 |
| `blue_policy_last_error` | 取得当前原生线程最近一次无法形成 JSON 响应的错误 |
| `blue_policy_shutdown` | 清空 session，并按解释器所有权执行最终关闭 |

## 2. Windows 基础交付环境

已验证组合：

- Windows x64；
- Visual Studio 2022、C++17；
- CPython 3.11 x64；
- NumPy 1.26.4；
- 模型运行时仅依赖 Python标准库和 NumPy，不需要 PyTorch、CUDA或绘图库。

当前基础交付采用“目标机准备兼容 Python 环境”的方式。目标机需安装 CPython 3.11 x64，
并安装：

```powershell
D:\TargetPython\python.exe -m pip install -r requirements-inference.txt
```

`python311.dll` 和 `zlib.dll` 不等于完整 Python 环境；标准库、NumPy及其原生依赖仍需由
`python_home_utf8` 指向的环境提供。完整私有 Python runtime 尚不属于本基础交付。

安装目录形状：

```text
dist/
├─ blue_policy_bridge.dll
├─ python311.dll
├─ zlib.dll
├─ requirements-inference.txt
├─ include/
│  └─ blue_policy_api.h
├─ lib/
│  └─ blue_policy_bridge.lib
├─ schemas/
│  ├─ blue_policy_request_v1.schema.json
│  └─ blue_policy_response_v1.schema.json
├─ blue_policy_test_host.exe
└─ python_app/
   ├─ blue_policy_entry.py
   ├─ blue_policy_json.py
   ├─ blue_policy_runtime/
   └─ model/
```

若 DLL 放在插件目录而不在宿主 EXE 目录，建议使用绝对路径和：

```cpp
LoadLibraryExW(
    absolute_dll_path,
    nullptr,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
```

这样系统在执行桥接代码前即可找到同目录的 `python311.dll`。不要依赖当前工作目录或把不受
信任目录加入全局 `PATH`。

## 3. 构建、安装与验证

在仓库根目录执行：

```powershell
Set-Location v1\deploy\blue_policy_dll
.\build.ps1 -PythonExecutable D:\Anaconda\python.exe -Install
```

脚本执行 Release x64 配置、编译、原生 CTest，并安装到 `dist/`。原生测试主机同时覆盖：

- API v2 JSON调用和响应释放；
- 旧 POD兼容路径；
- 多架飞机从多个 C++线程提交请求；
- JSON业务输出包含飞机编号和速度且不含动作序号。

Python回归测试：

```powershell
D:\Anaconda\python.exe -m pytest -q tests
```

## 4. 生命周期与调用顺序

一个宿主进程应遵循以下顺序：

```text
LoadLibraryExW（一次）
  → 检查 blue_policy_api_version()==2
  → blue_policy_initialize（一次）
  → JSON info（一次，核对模型、坐标、时序和能力）
  → 每架飞机 JSON reset（建立独立 session）
  → 各飞机连续 JSON step
  → 回合结束 close_session，或新回合再次 reset
  → 停止所有调用并等待工作线程退出
  → blue_policy_shutdown（一次）
  → FreeLibrary
```

初始化示例：

```cpp
BluePolicyInitConfig config{};
config.struct_size = sizeof(config);
config.python_home_utf8 = "D:/Anaconda";
config.python_module_dir_utf8 = "./python_app";
config.bundle_dir_utf8 = "./python_app";

if (blue_policy_initialize(&config) != BLUE_POLICY_STATUS_OK) {
    throw std::runtime_error(blue_policy_last_error());
}
```

如果桥接层创建了解释器，`blue_policy_shutdown` 必须在执行初始化的同一原生线程调用，并且
对该进程是终态。不要每帧或每回合重复初始化/关闭 CPython。

## 5. JSON 通用约定

- UTF-8，无 BOM；请求长度不包含结尾 `\0`。
- `protocol_version` 当前固定为整数 `1`。
- `request_id` 是非负 int64 或 1–64 字符字符串，响应原样回传。
- 输入严格拒绝未知字段、重复 key、`NaN/Infinity` 和嵌入 NUL。
- 最大请求 1 MiB、最大 JSON深度 32、最多 256 枚来弹、最多 1024 个 session。
- 原生返回码 `BLUE_POLICY_STATUS_OK` 只表示成功产生 JSON；仍须解析 JSON 的 `ok`。
- 原生返回非零时没有可用业务响应，应立即在同一线程复制
  `blue_policy_last_error()`，再按原生故障处理。
- 业务校验失败使用 `ok=false` 和稳定 `BPJ_*`错误对象，不把 Python异常类型暴露为接口。

## 6. `info`

初始化后首先发送：

```json
{
  "protocol_version": 1,
  "request_id": "startup-info",
  "op": "info"
}
```

调用方至少应核对：

- `dll_api_version=2`、`json_protocol_version=1`；
- capability 包含 `geodetic_input_v1`、`next_velocity_output_v1`、
  `multi_aircraft_concurrent_safe_v1`；
- `observation_dim=104`、`action_count=29`、`physics_step_s=0.005`；
- `velocity_output_semantics=next_physics_frame_velocity`；
- `concurrency_mode=serialized_gil_v1`；
- 模型和 Python bundle SHA-256。

## 7. 每架飞机建立独立 session

每架本机飞机使用唯一的非负 `aircraft_id` 和稳定 `session_id`。回合开始发送：

```json
{
  "protocol_version": 1,
  "request_id": "reset-101",
  "op": "reset",
  "session_id": "blue-101",
  "aircraft_id": 101,
  "aircraft_type": "fighter-a",
  "episode_id": 1,
  "input_mode": "exact"
}
```

成功响应：

```json
{
  "protocol_version": 1,
  "request_id": "reset-101",
  "ok": true,
  "result": {
    "operation": "reset",
    "session_id": "blue-101",
    "aircraft_id": 101,
    "aircraft_type": "fighter-a",
    "episode_id": 1,
    "input_mode": "exact",
    "reset": true
  }
}
```

一个 `aircraft_id` 同时只能绑定一个 session。再次 reset 同一个 session 会原子替换该飞机
的回合状态。`aircraft_type` 在当前版本是元数据和一致性校验字段，所有机型仍共享同一套
冻结模型和飞行包线；它尚不负责按机型选择不同模型。

`input_mode`：

- `exact`：每枚来弹必须提供 `energy` 和 `guidance_mode`；
- `kinematics_only`：允许省略这两个字段，由运行时根据运动学估计，需由项目方确认保真度。

## 8. `step` 完整输入

公共位置使用 WGS-84 经纬高，速度数组顺序固定为 `[北, 上, 东]`，单位为 `m/s`。经度、
纬度单位为度，高度单位为米。第一帧成功输入的本机经纬度用于建立该 session 的局部
北/上/东参考系，后续由适配层转换为模型内部坐标。

```json
{
  "protocol_version": 1,
  "request_id": "step-101-0",
  "op": "step",
  "session_id": "blue-101",
  "aircraft_id": 101,
  "aircraft_type": "fighter-a",
  "step_count": 0,
  "scenario_time_s": 0.0,
  "ownship": {
    "geodetic_position": {
      "longitude_deg": 108.0,
      "latitude_deg": 34.0,
      "altitude_m": 10000.0
    },
    "velocity_north_up_east_mps": [300.0, 0.0, 0.0],
    "alive": true,
    "actual_load_command_body_g": [0.0, 1.0, 0.0]
  },
  "incoming_missiles": [
    {
      "missile_id": 9001,
      "missile_type": "surface-to-air",
      "geodetic_position": {
        "longitude_deg": 108.0,
        "latitude_deg": 33.55,
        "altitude_m": 10000.0
      },
      "velocity_north_up_east_mps": [1000.0, 0.0, 0.0],
      "alive": true,
      "energy": 0.6,
      "guidance_mode": "locked"
    }
  ],
  "include_diagnostics": false
}
```

输入字段：

| 字段 | 要求 |
|---|---|
| `aircraft_id` | 本机唯一非负 int64；必须与 reset 绑定一致 |
| `aircraft_type` | 1–64 字符；必须与 reset 一致 |
| `step_count` | 非负 int64；同一 session 每帧严格加 1 |
| `scenario_time_s` | 当前场景时间；后续帧严格增加 `0.005 s`，容差 `1e-9 s` |
| `ownship.geodetic_position` | 本机 WGS-84 `longitude_deg/latitude_deg/altitude_m` |
| `ownship.velocity_north_up_east_mps` | 本机三轴速度 `[北,上,东]` |
| `ownship.alive` | 本机存活状态 |
| `actual_load_command_body_g` | 可选，`[轴向g,法向g,滚转rad]`；前两项范围 `[-10,10]`，滚转范围 `[-π,π]`；正式闭环建议传入真实执行值 |
| `incoming_missiles` | 来弹数组，可为空 |
| `missile_id` | 弹的稳定 int64 序号；单帧不得重复 |
| `missile_type` | 可选元数据 |
| 来弹位置/速度 | 与本机使用相同 WGS-84 和 `[北,上,东]`合同 |
| `alive` | 来弹存活状态 |
| `energy/guidance_mode` | `exact` 模式必填；能量 `[0,1]`，模式为 `boost/locked/lock_hold/inertial` |

当前模型要求本机高度位于 `8–12 km`，速度模长位于 `100–600 m/s`。来弹只有在与本机距离
严格小于 `60 km` 时进入内部观测，最多使用最近 3 枚；输入数组本身最多 256 枚。

## 9. `step` 输出：本机三轴速度

成功响应示例：

```json
{
  "protocol_version": 1,
  "request_id": "step-101-0",
  "ok": true,
  "result": {
    "operation": "step",
    "session_id": "blue-101",
    "aircraft_id": 101,
    "step_count": 0,
    "scenario_time_s": 0.0,
    "velocity_effective_time_s": 0.005,
    "next_velocity_north_up_east_mps": [300.0, 0.0, 0.0],
    "active": true,
    "fallback_required": false
  }
}
```

大项目使用 `next_velocity_north_up_east_mps` 作为下一物理帧的本机速度。该值由策略内部的
最终载荷、重力、速度/高度约束和 `0.005 s` 训练动力学推进得到，不是离散机动序号，也不是
长期目标速度。下一帧仍应把大项目实际执行后的经纬高和速度重新输入，避免宿主动力学与
策略内部单步预测的误差累积。

`velocity_effective_time_s` 表示该速度对应的场景时间。公共业务结果不会返回动作索引。

如果请求设置 `include_diagnostics=true`，响应可额外包含：运行模式、是否产生新决策、可见
来弹 ID、决策次数、安全干预和内部实际载荷。诊断对象同样不包含动作序号，生产环境建议
保持 `false`。

## 10. 关闭 session

```json
{
  "protocol_version": 1,
  "request_id": "close-101",
  "op": "close_session",
  "session_id": "blue-101",
  "aircraft_id": 101
}
```

关闭不存在的 session 返回 `closed=false`，因此可安全重复关闭。解释器关闭只能通过原生
`blue_policy_shutdown()` 执行，JSON协议没有 `shutdown` 操作。

## 11. 最小 C++ JSON 调用

完整示例见 [`examples/test_host.cpp`](examples/test_host.cpp)。响应内存必须由同一 DLL 的
`blue_policy_release_json` 释放：

```cpp
struct JsonResultOwner {
    BluePolicyJsonResult value{};

    JsonResultOwner() {
        value.struct_size = sizeof(value);
    }

    ~JsonResultOwner() {
        blue_policy_release_json(&value);
    }
};

std::string call_json(std::string_view request) {
    JsonResultOwner output;
    const int32_t status = blue_policy_call_json(
        request.data(), request.size(), &output.value);
    if (status != BLUE_POLICY_STATUS_OK) {
        const char* error = blue_policy_last_error();
        throw std::runtime_error(error == nullptr ? "unknown DLL error" : error);
    }
    return std::string(output.value.data_utf8, output.value.size_bytes);
}
```

调用前 `BluePolicyJsonResult` 必须清零并设置 `struct_size`。不得使用宿主的 `free/delete`
释放 `data_utf8`。释放函数对空结果和顺序重复释放安全；每次调用必须使用独立结果结构，
不得让多个线程同时释放同一个结果结构。

建议大项目再封装一个 RAII `BluePolicyClient`，统一处理 DLL加载、初始化、`info`核对、JSON
序列化、响应释放、业务错误、session映射和最终关闭，不要让业务模块散落调用 C函数。

## 12. 多机与多线程语义

- 多个 C++线程可以同时调用 `blue_policy_call_json`，不会破坏 DLL、Python或 session状态。
- 每架飞机有独立 `BluePolicySession`，独立保存威胁槽位、动作历史、估计器、飞行包线和
  时序状态。
- 冻结 Rainbow权重在 session之间只读共享；这不会共享各飞机的动态状态。
- 当前版本通过原生全局 mutex、CPython GIL 和每 session锁保证正确性，因此不同线程的
  推理在 DLL内部串行执行。它满足并发调用安全，但不承诺同时计算或吞吐加速。
- 同一 session 的 `step` 必须由调用方保证 FIFO；不要让两个线程竞争提交同一飞机的相邻帧。
- 不同飞机可以由不同 C++线程独立 reset/step/close。一个飞机请求失败不会修改其他飞机。
- 校验错误不提交 session状态；无法恢复的内部错误会把该 session标记为 tainted，并要求
  reset，不影响其他 session。

在当前“不要求性能、要求并发正确”的交付条件下，内部串行是刻意选择。若以后需要吞吐，
应在不改变 JSON接口的前提下增加 batch或多进程 worker。

## 13. 错误处理

必须区分两层错误：

1. 原生错误：空指针、未初始化、内存分配失败或 Python入口不可调用。C返回非零，并通过
   `blue_policy_last_error()` 读取诊断。
2. 业务错误：请求已被正常解析，但字段、时序或 session不合法。C返回 `OK`，JSON返回
   `ok=false`。

业务错误响应示例：

```json
{
  "protocol_version": 1,
  "request_id": "step-101-1",
  "ok": false,
  "error": {
    "code": "BPJ_SEQUENCE_ERROR",
    "message": "step_count and scenario_time_s must advance by one physics frame",
    "path": "$.step_count",
    "retryable": false,
    "state_changed": false,
    "reset_required": false,
    "details": {
      "expected_step_count": 1,
      "expected_scenario_time_s": 0.005
    }
  }
}
```

调用方必须先检查原生返回码，再解析 JSON，最后检查 `ok`。当
`reset_required=true` 时不得继续 step，应对对应飞机重新 reset。

## 14. 接续开发与可修改边界

下列内容保持外置，可在不改变 C ABI、无需重新编译 DLL 的情况下修改：

- `python_app/blue_policy_json.py` 中的输入适配、严格校验、经纬高转换和速度后处理；
- `python_app/blue_policy_runtime/*.py` 中的观测构造、推理前后处理、威胁状态和飞行包线；
- `python_app/model/*.npz` 模型权重及 manifest，但新权重必须满足当前模型合同并更新哈希；
- 根据已有 `aircraft_type` 字段在 Python内部增加机型到模型/参数的映射。

不重新编译的前提是以下公共合同保持不变：

- C导出符号、参数、调用约定和响应内存所有权；
- Python入口仍提供 `initialize/dispatch_json/shutdown`，并保留 DLL 加载时要求的
  `predict/create_session/reset_session/step/destroy_session` 兼容 callable；
- JSON `protocol_version=1` 的字段名称、类型、单位、坐标顺序和时序语义；
- 公共 step输出仍是 `aircraft_id + next_velocity_north_up_east_mps`；
- 当前模型合同仍为 104维观测、29个内部动作和61个 atoms；
- CPython 3.11 ABI和 NumPy依赖保持兼容。

更新流程：修改 Python/模型 → 运行单元和 golden test → 更新模型/bundle哈希 → 停止宿主
→ 替换 `python_app` → 重启宿主 → 调用 `info`自检。当前不支持进程内热 reload。

以下修改必须重新编译并重新验收 DLL：

- 修改 C ABI、导出符号、结构体布局、调用约定或内存所有权；
- CPython 3.11升级到其他次版本 ABI系列；
- x64/Windows目标或原生依赖变化；
- 删除或改变现有 JSON字段的类型、单位或含义时，需要同时提升协议版本。

输入端严格拒绝未知字段。若只想新增可选请求字段，应先通过 `info.capabilities` 协商；不能
假设旧 dispatcher 会自动忽略它。

## 15. 旧 POD 兼容接口

下列 API v1风格接口仍保留在 API v2 DLL 中：

- `blue_policy_predict(float[104])`；
- `blue_policy_create_session/reset_session/step/destroy_session`；
- `BluePolicyFrame/BluePolicyOutput`。

它们仍可能返回内部动作序号和载荷，只为旧调用方和一致性测试保留。新业务不得以其
`action_index` 作为控制输出，应使用 JSON接口返回的三轴速度。JSON session和 POD session
属于两套独立注册表，不要为同一架飞机混合使用两种状态接口。

## 16. 当前限制

- 仅完成并验证 Windows x64基础交付；本阶段不承诺跨平台。
- 多线程调用安全，但单 CPython解释器内部串行，不是并行计算加速。
- `aircraft_type` 当前只做元数据/一致性校验，不会自动选择不同机型模型或动力学参数。
- 所有飞机共享同一份只读 Rainbow权重，但各自状态完全隔离。
- 输出速度是下一 `0.005 s` 物理帧速度，不是长期航迹、航点或期望速度闭环。
- 当前交付依赖目标机的 CPython 3.11+NumPy环境，不是完全免安装私有 runtime。
- 动态库不是硬实时保证、安全沙箱或源码保护；正式接入仍需在目标机做长时间和故障测试。
