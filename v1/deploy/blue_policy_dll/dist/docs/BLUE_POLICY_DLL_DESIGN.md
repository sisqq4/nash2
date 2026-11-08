# 蓝方推理动态链接库 API v2 设计与集成规范

> 文档状态：已实现的基础交付基线  
> 最后更新：2026-11-07  
> 原生接口：DLL API v2  
> JSON 协议：protocol v1  
> 当前验收平台：Windows x64  
> 当前范围：蓝方在线推理；不包含训练、评估绘图和图像输出

## 1. 工作结论与当前状态

本项目已经按下列路径完成蓝方推理的基础 DLL 封装：

```text
大项目中的 C++ 调用方
  → 稳定 C ABI：blue_policy_call_json()
  → CPython 3.11 C API 桥接
  → Python 严格 JSON 协议层
  → 经纬高/速度/来弹状态转内部 104 维观测
  → 冻结 Rainbow C51 推理和飞行包线
  → 单步动力学计算下一帧速度
  → C++ 获得本机编号和北/上/东三轴速度
```

当前实现已经满足本阶段提出的基础要求：

| 需求 | 当前实现 |
| --- | --- |
| C++ 输入较少的物理字段 | 已实现；调用方不需要构造内部 104 维观测 |
| 输入本机编号、时间、机型、经纬高和三轴速度 | 已实现 |
| 输入来弹序列、来弹编号、经纬高和三轴速度 | 已实现 |
| 保留算法已有的必要输入 | 已实现；包括 session、回合、步数、存活状态、输入模式和可选实际载荷反馈等 |
| 输出本机编号和三轴速度 | 已实现 |
| 正式输出不再使用机动/动作序号 | 已实现；JSON 业务响应不含 `action_index` 或 `raw_action_index` |
| 多架飞机分别维护推理状态 | 已实现；每架飞机一个独立 session |
| C++ 多线程并发调用安全 | 已实现；当前 DLL 内部串行，优先保证正确性 |
| Python 算法修改后不重编译 DLL | 已实现；替换 Python/模型文件并重启宿主即可 |
| 调整 C++ 对接字段较方便 | 已实现；稳定 C ABI 只传 UTF-8 JSON，业务字段由 Python 协议层处理 |
| 跨平台 | 本阶段未验收 |
| 性能优化或真正同时计算 | 本阶段未要求；当前调用安全但内部串行 |

这里所说的“每架飞机一个网络”，实现含义是：每架飞机拥有独立的威胁槽位、估计器、动作
历史、时序和飞行包线状态；相同的冻结模型权重只读共享。复制完全相同的权重不会改变推理
结果，只会增加内存，因此当前没有为每架飞机重复加载一份权重。

## 2. 封装范围

### 2.1 已包含

- 冻结 Rainbow 模型前向推理；
- 小型物理输入到内部 `normalized_v4` 104 维观测的构造；
- WGS-84 经纬高与模型局部坐标的转换；
- 来弹筛选、稳定槽位、运动学估计和跨帧状态；
- 飞行包线、安全干预和载荷插值；
- 一帧 `0.005 s` 的飞机动力学推进；
- 严格 JSON 校验、稳定错误码和机器可读 Schema；
- 多飞机 session 管理和 C++ 多线程安全调用；
- CPython 生命周期、GIL、异常和响应内存管理；
- Windows x64 Release 构建、安装包和原生测试主机。

### 2.2 不包含

- 训练、续训、梯度、优化器、经验回放和目标网络；
- 训练或评估场景生成；
- Matplotlib、图像、视频、ACMI 或报告输出；
- 红方策略或红方控制；
- 大项目自己的全局飞机注册、任务分配和仿真调度；
- 完整的私有 Python 运行时镜像；
- Linux/macOS 交付验证；
- 真正的多解释器或多进程并行推理；
- 运行中的 Python 热更新。

## 3. 代码和交付物位置

主要实现文件如下：

| 内容 | 文件 |
| --- | --- |
| 公共 C ABI | `v1/deploy/blue_policy_dll/include/blue_policy_api.h` |
| CPython C API 桥接 | `v1/deploy/blue_policy_dll/src/blue_policy_bridge.cpp` |
| 强类型 C++17 客户端 | `v1/deploy/blue_policy_dll/include/blue_policy_client.hpp`、`src/blue_policy_client.cpp` |
| Python 固定入口 | `v1/deploy/blue_policy_dll/python_app/blue_policy_entry.py` |
| JSON、坐标、校验与多机注册表 | `v1/deploy/blue_policy_dll/python_app/blue_policy_json.py` |
| 推理运行时 | `v1/deploy/blue_policy_exe/blue_policy_runtime/` |
| 冻结模型与 manifest | `v1/deploy/blue_policy_exe/model/` |
| 请求 Schema | `v1/docs/schemas/blue_policy_request_v1.schema.json` |
| 响应 Schema | `v1/docs/schemas/blue_policy_response_v1.schema.json` |
| C++ 调用示例 | `v1/deploy/blue_policy_dll/examples/blue_policy_client_example.cpp` |
| C++ 底层与强类型测试 | `examples/test_host.cpp`、`tests/test_cpp_client.cpp` |
| Python/DLL 回归测试 | `v1/deploy/blue_policy_dll/tests/` |
| 面向使用者的快速说明 | `v1/deploy/blue_policy_dll/README.md` |

安装后的目录为：

```text
dist/
├─ blue_policy_bridge.dll
├─ python311.dll
├─ zlib.dll
├─ requirements-inference.txt
├─ blue_policy_test_host.exe
├─ blue_policy_client_example.exe
├─ blue_policy_client_test.exe
├─ include/
│  ├─ blue_policy_api.h
│  └─ blue_policy_client.hpp
├─ lib/
│  ├─ blue_policy_bridge.lib
│  └─ blue_policy_client.lib
├─ schemas/
│  ├─ blue_policy_request_v1.schema.json
│  └─ blue_policy_response_v1.schema.json
└─ python_app/
   ├─ blue_policy_entry.py
   ├─ blue_policy_json.py
   ├─ blue_policy_runtime/
   └─ model/
```

`python311.dll` 和 `zlib.dll` 并不构成完整 Python 环境；目标机仍须提供 Python 标准库、
NumPy 及其原生依赖。

## 4. 总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│ C++ BluePolicyClient                                        │
│ DLL 加载 / 初始化 / JSON 序列化 / session 映射 / 错误处理   │
└──────────────────────────────┬───────────────────────────────┘
                               │ UTF-8 JSON
┌──────────────────────────────▼───────────────────────────────┐
│ blue_policy_bridge.dll                                      │
│ C ABI / 参数检查 / 全局 mutex / GIL / CPython / 内存所有权   │
└──────────────────────────────┬───────────────────────────────┘
                               │ dispatch_json(str) -> str
┌──────────────────────────────▼───────────────────────────────┐
│ blue_policy_json.py                                         │
│ 严格解析 / 稳定错误 / WGS-84 转换 / session 与飞机身份绑定   │
└──────────────────────────────┬───────────────────────────────┘
                               │ 小型、命名的物理帧
┌──────────────────────────────▼───────────────────────────────┐
│ BluePolicySession                                           │
│ 威胁状态 / 104 维观测 / Rainbow / 飞行包线 / 载荷插值        │
└──────────────────────────────┬───────────────────────────────┘
                               │ 内部最终载荷
┌──────────────────────────────▼───────────────────────────────┐
│ 单步训练动力学                                               │
│ 重力 + 载荷 + 速度/高度约束 → 下一帧 N/U/E 速度              │
└──────────────────────────────────────────────────────────────┘
```

选择“固定 C ABI + JSON 业务协议”的主要原因是：

- C ABI 不暴露 STL、Python 对象或 C++ 异常，跨编译单元更稳定；
- C++ 不需要与 Python 的字典、NumPy 内存布局或内部 104 维观测耦合；
- 输入字段、校验、机型路由和算法后处理可以在 Python 修改；
- JSON protocol 可以独立版本化；
- Schema 可同时用于文档、调用方代码生成和自动测试。

## 5. 坐标、单位与时间合同

### 5.1 公共坐标

公共 JSON 接口使用：

- 位置：WGS-84 `longitude_deg`、`latitude_deg`、`altitude_m`；
- 速度：`velocity_north_up_east_mps = [north, up, east]`；
- 距离和高度：米；
- 速度：米/秒；
- 时间：秒。

不要把速度数组理解为 `[x,y,z]`、`[east,north,up]` 或经纬高变化率。

### 5.2 内部固定切平面

每个 session 在第一帧成功 `step` 时，以本机经纬度建立固定参考原点，后续不随飞机重新
定心：

1. WGS-84 经纬高转换到 ECEF；
2. 位置投影到该 session 固定的北/东切平面；
3. 模型内部轴保持 `x=北、y=上、z=东`；
4. 内部 `y` 使用绝对高度，保持训练时 `8–12 km` 的语义；
5. 各实体自身位置的 N/U/E 速度先转 ECEF，再转到固定参考系；
6. 输出速度再转回本机当前位置的 N/U/E 基底。

固定原点可以避免每帧重新定心导致坐标轴旋转和跨帧航迹不连续。

### 5.3 时序

- `scenario_time_s` 是当前输入帧的场景时间；
- `step_count` 是当前输入物理帧序号；
- reset 后第一帧可以从任意非负 `step_count` 和非负时间开始；
- 第一帧成功后，后续必须严格 `step_count + 1`；
- 后续时间必须严格增加 `0.005 s`，允许绝对误差 `1e-9 s`；
- 输出 `velocity_effective_time_s = scenario_time_s + 0.005`。

同一飞机不得由两个线程竞争提交相邻帧。全局锁只能防止内存竞争，不能替调用方决定两个
并发请求的业务先后顺序。

## 6. 为什么 C++ 不输入 104 维

104 维是当前 Rainbow 模型的内部观测，不是公共接口。Python 运行时根据物理输入自动生成：

```text
6 维本机量
+ 3 个来弹槽 × 22 维威胁编码
+ 29 维上一次内部动作 one-hot
+ 3 维实际载荷反馈
= 104 维
```

其中威胁槽位、能量/制导估计、归一化、上一次动作和载荷历史都具有跨帧状态。若让 C++ 直接
构造 104 维，会把算法内部布局永久泄漏到大项目，也容易造成训练与部署不一致。因此 C++
只提交本机和来弹的命名物理字段。

## 7. C ABI 设计

公共头文件中的 API 版本为：

```cpp
#define BLUE_POLICY_API_VERSION 2u
```

首选业务接口：

```cpp
typedef struct BluePolicyJsonResult {
    uint32_t struct_size;
    char* data_utf8;
    size_t size_bytes;
} BluePolicyJsonResult;

int32_t blue_policy_call_json(
    const char* request_utf8,
    size_t request_size_bytes,
    BluePolicyJsonResult* out_result);

void blue_policy_release_json(BluePolicyJsonResult* result);
```

### 7.1 响应内存规则

- 调用前把结果结构清零，再设置 `struct_size=sizeof(BluePolicyJsonResult)`；
- `request_size_bytes` 不包含调用方可能附加的结尾 `\0`；
- 成功时 DLL 分配 `data_utf8`，并额外附加一个不计入 `size_bytes` 的结尾 NUL；
- 必须用同一个已加载 DLL 的 `blue_policy_release_json()` 释放；
- 禁止使用宿主的 `free`、`delete` 或其他 CRT 释放；
- release 后指针和长度会清零，顺序重复 release 安全；
- 每个在途调用使用独立结果结构；不得让两个线程同时释放同一结构；
- 结果结构非空时再次调用会返回原生参数错误，不会隐式覆盖旧指针。

不能使用“第一次调用查询长度、第二次调用取得结果”的两遍模式，因为 `step` 是有状态操作，
重复请求可能推进两次 session。

### 7.2 两层返回状态

必须区分：

1. C 返回码：桥接层是否成功形成一份 JSON 响应；
2. JSON `ok`：业务请求是否通过校验并执行成功。

因此，C 返回 `BLUE_POLICY_STATUS_OK` 时仍然必须解析 JSON 并检查 `ok`。非法 UTF-8 字节、
嵌入 NUL、未初始化、空指针和内存失败等无法安全形成业务 JSON 的情况返回原生非零状态，
诊断通过当前线程的 `blue_policy_last_error()` 读取并立即复制。

### 7.3 保留的旧 POD 接口

API v2 DLL 仍保留：

- `blue_policy_predict(float[104])`；
- `blue_policy_create_session/reset_session/step/destroy_session`；
- `BluePolicyFrame/BluePolicyOutput`。

这些接口仅用于旧调用方兼容、底层诊断和 golden test。它们仍暴露内部动作序号和载荷，不是
本次新业务输出合同。新大项目必须使用 JSON 接口，不要把 POD 与 JSON session 混用。

## 8. JSON 通用合同

所有请求均包含：

```json
{
  "protocol_version": 1,
  "request_id": "caller-generated-id",
  "op": "info"
}
```

通用规则：

- UTF-8，无 BOM；
- `protocol_version` 固定为整数 `1`；
- `request_id` 是非负 int64 或 1–64 字符字符串；
- 最大请求 1 MiB；
- 最大 JSON 嵌套深度 32；
- 拒绝未知字段、重复 key、`NaN`、`Infinity`、错误类型和非有限数；
- 最多 256 枚输入来弹；
- 最多 1024 个 session；
- `session_id` 长度 1–128，只允许 `[A-Za-z0-9_.:-]`；
- 成功响应原样回传 `request_id`；无法可靠读取 ID 的解析错误回传 `null`。

Schema：

- `v1/docs/schemas/blue_policy_request_v1.schema.json`
- `v1/docs/schemas/blue_policy_response_v1.schema.json`

目标机运行 DLL 不需要安装 `jsonschema`；Schema 主要供调用方、文档和开发测试使用。

## 9. `info`：启动自检

初始化后首先调用：

```json
{
  "protocol_version": 1,
  "request_id": "startup-info",
  "op": "info"
}
```

调用方至少核对：

- `dll_api_version == 2`；
- `json_protocol_version == 1`；
- capabilities 包含 `geodetic_input_v1`、`next_velocity_output_v1` 和
  `multi_aircraft_concurrent_safe_v1`；
- `observation_dim == 104`、`action_count == 29`、`threat_slot_count == 3`；
- `physics_step_s == 0.005`、`decision_interval_s == 0.1`；
- `velocity_output_semantics == "next_physics_frame_velocity"`；
- `concurrency_mode == "serialized_gil_v1"`；
- 模型 SHA-256 和 Python bundle SHA-256 是否为项目允许版本；
- Python、NumPy、架构和解释器所有权是否符合部署配置。

Python bundle SHA-256 覆盖交付目录中的 Python 文件和模型 manifest；模型 `.npz` 另由
`model_sha256` 校验。

## 10. `reset`：绑定一架本机飞机

每架本机飞机必须使用唯一 `aircraft_id` 和稳定 `session_id`：

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

规则：

- 一个 `aircraft_id` 同时只能绑定一个 session；
- `aircraft_id` 是本机编号，不在输出中列出其他友机；
- `aircraft_type` 在当前版本用于元数据和一致性校验；
- 后续 step 的飞机编号和类型必须与 reset 完全一致；
- 同一 session 再次 reset 会创建干净的新回合状态；
- `input_mode=exact` 要求每枚来弹提供能量和制导模式；
- `input_mode=kinematics_only` 允许只提供运动学，由运行时估计缺失量。

当前 `aircraft_type` 不会选择不同模型。后续可以在 Python 内扩展为
`aircraft_type -> model/config` 注册表，而不改变 C ABI。

## 11. `step`：完整输入合同

推荐的完整 exact 请求：

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

### 11.1 本机字段

| 字段 | 类型与语义 |
| --- | --- |
| `session_id` | reset 时建立的 session |
| `aircraft_id` | 本机非负 int64 编号 |
| `aircraft_type` | 1–64 字符，必须与 reset 一致 |
| `step_count` | 当前物理帧非负 int64 序号 |
| `scenario_time_s` | 当前场景时间，有限且非负 |
| `geodetic_position` | WGS-84 经度、纬度、高度 |
| `velocity_north_up_east_mps` | `[北,上,东]` 三轴速度 |
| `alive` | 本机存活状态 |
| `actual_load_command_body_g` | 可选的真实执行反馈 `[轴向 g, 法向 g, 滚转 rad]` |

本机约束：

- 经度 `[-180,180]` 度，纬度 `[-90,90]` 度；
- 当前模型高度 `8,000–12,000 m`；
- 当前模型速度模长 `100–600 m/s`；
- 实际载荷前两项范围 `[-10,10] g`，滚转范围 `[-π,π] rad`；
- 若省略实际载荷，运行时回退到上一帧由库产生的内部载荷；若大项目未完全执行库的控制，
  建议传入真实反馈以保持观测一致。

### 11.2 来弹字段

| 字段 | 类型与语义 |
| --- | --- |
| `missile_id` | 稳定 int64 编号；单帧不得重复 |
| `missile_type` | 可选的 1–64 字符元数据 |
| `geodetic_position` | 来弹 WGS-84 经纬高 |
| `velocity_north_up_east_mps` | 来弹 `[北,上,东]` 三轴速度 |
| `alive` | 来弹存活状态 |
| `energy` | `exact` 必填，范围 `[0,1]` |
| `guidance_mode` | `exact` 必填：`boost/locked/lock_hold/inertial` |

数组顺序用于传输当前来弹序列，但身份只由稳定 `missile_id` 决定，不能用数组下标替代 ID。
运行时从距离严格小于 `60 km` 的来弹中使用最近 3 枚，最多仍允许调用方提交 256 枚已知航迹。

## 12. `step` 输出合同

成功响应示例中的速度数值仅作形状说明：

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
    "next_velocity_north_up_east_mps": [299.999, 0.012, -0.021],
    "active": true,
    "fallback_required": false
  }
}
```

正式业务输出的控制量是：

```text
aircraft_id
+ next_velocity_north_up_east_mps[3]
```

不再返回机动序号、原始动作序号或目标动作序号。内部离散动作仍然存在，因为 Rainbow、动作
历史和载荷插值需要它，但它不会越过新的业务接口。

### 12.1 下一帧速度的含义

`next_velocity_north_up_east_mps` 不是当前速度、长期目标速度或简单改名后的动作序号。它由：

```text
当前速度
  → 内部最终载荷
  → 随速度坐标系中的加速度
  → 加重力
  → 用 dt=0.005 s 积分
  → 限制到 100–600 m/s
  → 处理 8–12 km 高度边界
  → 转回本机当前位置的 N/U/E 速度
```

大项目若采用这个速度作为下一帧速度，就不要再对同一帧重复施加诊断中的内部载荷，否则会
重复积分。下一帧继续把大项目真实执行后的经纬高和速度送回 DLL。

### 12.2 诊断输出

`include_diagnostics=true` 时可返回：

- 当前模式；
- 是否在本帧更新策略决策；
- 可见来弹 ID；
- 决策次数；
- 飞行包线/安全干预状态；
- 内部实际载荷。

诊断对象也不包含动作序号。生产调用建议保持 `false`。

## 13. `close_session`

```json
{
  "protocol_version": 1,
  "request_id": "close-101",
  "op": "close_session",
  "session_id": "blue-101",
  "aircraft_id": 101
}
```

成功时 `closed=true`；session 已不存在时 `closed=false`，因此可顺序重复关闭。JSON 协议不
提供进程级 `shutdown`，最终关闭必须调用原生 `blue_policy_shutdown()`。

## 14. 输入校验、错误和状态原子性

### 14.1 桥接层校验

C++ 层负责：

- 初始化状态、空指针和 `struct_size`；
- 请求长度 `1..1,048,576`；
- 嵌入 NUL；
- 严格 UTF-8 解码；
- Python callable 和返回类型；
- 最大 4 MiB 响应；
- 内存分配和 C ABI 异常隔离。

这类失败返回 `BluePolicyStatus` 非零值，并通过 `blue_policy_last_error()` 诊断。

### 14.2 Python 协议层校验

业务层负责：

- JSON 语法、重复 key、深度、非法常量和未知字段；
- 字段类型、长度、范围和有限数；
- 经纬高、速度、载荷和来弹唯一性；
- aircraft/session 绑定；
- step 和时间连续性；
- exact 模式的能量和制导字段；
- session 数量和来弹数量限制。

稳定业务错误码包括：

| 错误码 | 含义 |
| --- | --- |
| `BPJ_INVALID_UTF8` | BOM 或文本编码合同不满足 |
| `BPJ_INVALID_JSON` | JSON 语法、重复 key 或非法常量 |
| `BPJ_UNSUPPORTED_PROTOCOL` | 协议版本不支持 |
| `BPJ_UNSUPPORTED_OPERATION` | `op` 不支持 |
| `BPJ_MISSING_FIELD` | 缺少必填字段 |
| `BPJ_UNKNOWN_FIELD` | 存在未声明字段 |
| `BPJ_TYPE_MISMATCH` | JSON 类型或三元组形状错误 |
| `BPJ_OUT_OF_RANGE` | 数值、字符串、数量或物理范围错误 |
| `BPJ_DUPLICATE_TRACK_ID` | 单帧来弹 ID 重复 |
| `BPJ_SESSION_NOT_FOUND` | 尚未 reset 或已关闭 |
| `BPJ_SEQUENCE_ERROR` | 步数或时间不连续 |
| `BPJ_AIRCRAFT_IN_USE` | 本机编号已经绑定其他 session |
| `BPJ_AIRCRAFT_MISMATCH` | 本机编号或类型与 session 不一致 |
| `BPJ_SESSION_TAINTED` | session 无法保证状态，需要 reset |
| `BPJ_INTERNAL_ERROR` | 算法运行时未预期失败 |

业务错误响应统一为：

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

### 14.3 原子性

所有协议字段在进入算法前完整校验。算法 step 前会保存 session 快照；若算法 step 失败，
运行时恢复到快照，所以调用方可以修正同一帧后重试。只有无法可靠保存/恢复状态时才标记
session tainted，并通过 `reset_required=true` 要求重建该飞机的 session。一个 session 的
失败不会修改其他飞机。

## 15. 多飞机与并发设计

当前并发语义是“concurrent-safe / internally serialized”：

- 多个 C++ 线程可以同时进入 `blue_policy_call_json()`；
- 原生全局 mutex 覆盖完整调用，CPython 调用同时受 GIL 保护；
- 不同飞机拥有不同 `BluePolicySession`，动态状态不共享；
- 相同的 NumPy 模型数组被设为只读并在 session 之间共享；
- 每次响应使用独立的 DLL-owned buffer；
- reset、step、close 和 shutdown 不会在内部发生未保护的数据竞争；
- 当前不同飞机的推理仍在 DLL 内排队，不承诺同时计算或吞吐提升。

推荐的大项目结构：

```text
飞机 101 仿真线程 ─┐
飞机 102 仿真线程 ─┼─→ BluePolicyClient ─→ DLL 内部串行 ─→ 各自 session
飞机 103 仿真线程 ─┘
```

调用方必须遵守：

- 每架飞机使用唯一 session；
- 同一飞机只允许一个 FIFO 提交者；
- 不同飞机可以由不同线程调用；
- 每个线程/调用使用独立 request、response 和结果结构；
- shutdown 前停止新请求并 join 全部调用线程；
- 不要在 step 正在排队时从另一个线程随意 close 同一 session。

用户当前不要求性能，因此内部串行是基础交付中更稳妥的方案。若以后需要真正并行，可在不
改变 JSON 合同的情况下增加多进程 worker 或按飞机分片的多个 DLL 宿主进程。

## 16. C++ 大项目对接方案

### 16.1 生命周期

```text
LoadLibraryExW 或正常链接
  → blue_policy_api_version()==2
  → blue_policy_initialize()
  → JSON info 并核对 capability/哈希
  → 每架飞机 reset
  → 每 0.005 s 对各飞机 step
  → 回合结束 close_session 或重新 reset
  → 停止请求并等待所有工作线程
  → blue_policy_shutdown()
  → FreeLibrary
```

不要每帧加载 DLL、每回合初始化 CPython，或者在推理线程仍运行时卸载 DLL。

### 16.2 初始化

```cpp
#include "blue_policy_api.h"

BluePolicyInitConfig config{};
config.struct_size = sizeof(config);
config.python_home_utf8 = "D:/Anaconda";
config.python_module_dir_utf8 = "D:/app/blue_policy/python_app";
config.bundle_dir_utf8 = "D:/app/blue_policy/python_app";

const int32_t status = blue_policy_initialize(&config);
if (status != BLUE_POLICY_STATUS_OK) {
    throw std::runtime_error(blue_policy_last_error());
}
```

如果宿主已经初始化 CPython，DLL 会借用它；否则 DLL 自己创建解释器。DLL 自己创建解释器
时，`blue_policy_shutdown()` 必须在初始化的同一原生线程调用，而且完整 finalize 后该进程
不能再次用本 DLL 创建解释器。

### 16.3 RAII JSON 调用

```cpp
#include <stdexcept>
#include <string>
#include <string_view>

struct JsonResultOwner {
    BluePolicyJsonResult value{};

    JsonResultOwner() {
        value.struct_size = sizeof(value);
    }

    ~JsonResultOwner() {
        blue_policy_release_json(&value);
    }
};

std::string CallBluePolicy(std::string_view request) {
    JsonResultOwner result;
    const int32_t status = blue_policy_call_json(
        request.data(), request.size(), &result.value);
    if (status != BLUE_POLICY_STATUS_OK) {
        const char* error = blue_policy_last_error();
        throw std::runtime_error(error == nullptr ? "DLL error" : error);
    }
    return std::string(result.value.data_utf8, result.value.size_bytes);
}
```

随后由大项目已有 JSON 库解析字符串，先检查 `ok`，再读取 `result.aircraft_id` 和
`result.next_velocity_north_up_east_mps`。DLL 本身不要求 C++ 链接特定 JSON 库。

### 16.4 建议封装的 `BluePolicyClient`

大项目应只保留一个统一 client，负责：

- DLL 加载、符号解析、API 版本和 `info` 核对；
- CPython 初始化与最终关闭；
- `aircraft_id -> session_id` 映射；
- request ID、JSON 序列化和响应释放；
- 原生错误与业务错误分层；
- 单机 FIFO 和超时/降级策略；
- 日志中的模型哈希、飞机、回合、步数和错误码。

业务模块不要散落调用原生 C 函数。

## 17. 环境与目标机部署

### 17.1 当前已验证组合

| 项目 | 版本/要求 |
| --- | --- |
| 操作系统 | Windows x64 |
| 编译器 | Visual Studio 2022，C++17 |
| 构建工具 | CMake 3.18+ |
| Python | CPython 3.11.14 x64 |
| NumPy | 1.26.4 |
| 模型运行依赖 | Python 标准库 + NumPy |
| 不需要 | PyTorch、CUDA、Matplotlib、Pillow |

目标机安装依赖：

```powershell
D:\TargetPython\python.exe -m pip install -r requirements-inference.txt
```

当前 CMake 精确查找 Python 3.11 的 Embed Development 库。升级到 Python 3.12/3.13、改变
位数或调用约定，需要重新编译和验收 DLL。

### 17.2 DLL 搜索

推荐用绝对路径和 `LoadLibraryExW`：

```cpp
LoadLibraryExW(
    absolute_dll_path,
    nullptr,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
```

这样 Windows 可以在执行桥接代码前找到 DLL 同目录的 `python311.dll`。不要依赖当前工作
目录，也不要把不受信任目录加入全局 `PATH`。

### 17.3 完整私有 runtime

未来可以把 CPython、标准库、NumPy 和原生依赖完整放入应用私有目录，使对方机器不需要
全局安装 Python。但应用内部仍然包含 Python runtime；这项工作尚未完成，当前基础交付要求
目标机准备兼容的 CPython 3.11 + NumPy 环境。

## 18. Python 和模型的接续开发

### 18.1 无需重新编译 DLL 的修改

下列修改通常只需替换文件并重启宿主：

- `blue_policy_json.py` 的字段适配、校验、坐标和输出后处理；
- `blue_policy_runtime/*.py` 的观测构造、威胁状态、飞行包线和推理前后处理；
- 满足当前合同的新模型 `.npz` 和更新后的 manifest；
- 在 Python 内新增 `aircraft_type -> model/config` 路由；
- 向响应增加调用方会忽略的兼容性诊断字段。

前提是：

- C 导出符号、调用约定和结果内存合同不变；
- `blue_policy_entry.py` 继续提供 DLL 加载时要求的
  `initialize/dispatch_json/predict/create_session/reset_session/step/destroy_session/shutdown`；
- JSON v1 字段名称、类型、单位、坐标顺序和时序含义不变；
- 正式 step 输出仍是本机编号和下一帧 N/U/E 速度；
- 当前模型合同仍满足 104 维观测、29 个内部动作、61 个 atoms 和 3 个威胁槽；
- CPython 3.11 ABI 和 NumPy 原生依赖仍兼容。

推荐更新流程：

```text
修改 Python/模型
  → 运行 Python、Schema、golden 和原生测试
  → 更新模型 manifest 和哈希
  → 停止宿主
  → 替换 python_app
  → 重启宿主
  → 发送 info 核对版本和哈希
```

当前不支持进程内热 reload。Python 会缓存模块，既有 session 也持有旧类实例；直接覆盖
正在运行的 `.py` 文件不会安全地改变当前对象。

### 18.2 必须重新编译或提升协议的修改

以下情况需要重新编译 DLL：

- 修改 C 导出函数、参数、结构体布局、调用约定或内存所有权；
- 更换 CPython ABI 系列；
- 更换目标架构或原生依赖策略；
- 改成 Limited API、动态解析 CPython 或其他原生桥接路线。

删除字段、改变类型/单位/含义属于破坏性 JSON 变化，必须提升 JSON 协议版本并同步修改调用
方、Schema 和测试。新增可选请求字段也需要 capability 协商，因为当前严格解析器会拒绝未知
字段。

## 19. 构建和验证

构建、原生测试并安装：

```powershell
Set-Location v1\deploy\blue_policy_dll
.\build.ps1 -PythonExecutable D:\Anaconda\python.exe -Install
```

Python 和 borrowed-interpreter DLL 回归：

```powershell
$env:BLUE_POLICY_DLL = (Resolve-Path .\dist\blue_policy_bridge.dll).Path
D:\Anaconda\python.exe -m pytest -q tests
```

截至 2026-11-07，本次实际验证结果：

- Release x64 DLL 编译成功；
- CTest 原生测试：`2/2 passed`；
- DLL 目录安装成功；
- DLL/Python 测试：`14 passed`；
- 原生测试覆盖 8 个 C++ 线程、8 架不同飞机，每架连续多帧；
- 强类型 C++ 客户端测试覆盖完整生命周期、字符串转义、int64精度、业务错误和并发调用；
- borrowed-interpreter 测试覆盖 `info → reset → step → close → shutdown → reopen`；
- Schema 校验覆盖实际 info/reset/step/close 响应；
- 两份 Markdown 中的所有 JSON 示例均由自动测试按请求/响应 Schema 校验；
- 验证正式 step 响应及 diagnostics 不含动作序号；
- 验证错误帧不消耗时序、响应内存可释放并可顺序重复释放。

## 20. 当前仍需后续完成的内容

当前代码已经达到“可供大项目开始集成”的基础交付状态，但生产交付前仍建议完成：

1. 在对方目标机的真实 Python、DLL 搜索路径和部署目录中做一次冷启动验收；
2. 确定目标机采用“预装 Python”还是完整私有 Python runtime；
3. 与大项目确认下一帧速度的采用方式，避免再重复施加载荷；
4. 进行长时间、多回合、异常请求和 shutdown 压力测试；
5. 明确大项目在 `fallback_required`、超时和 session tainted 时的安全降级速度；
6. 若不同 `aircraft_type` 确实对应不同模型或动力学，补充机型路由和相应模型包；
7. 如未来要求吞吐，再设计多进程或 batch；当前不需要为性能破坏稳定接口；
8. 跨平台工作待需求明确后另行实施和验收。

## 21. 大项目接入检查清单

- [ ] 只使用 API v2 JSON 作为新业务接口；
- [ ] 启动时核对原生 API、JSON protocol、capabilities 和哈希；
- [ ] 每架飞机唯一 `aircraft_id/session_id`，先 reset 再 step；
- [ ] 速度顺序固定为 `[北,上,东]`；
- [ ] 经纬高为 WGS-84 度/米；
- [ ] 来弹 ID 跨帧稳定，单帧不重复；
- [ ] 同一飞机请求严格 FIFO，时间每帧增加 `0.005 s`；
- [ ] 每次调用使用独立且清零的 `BluePolicyJsonResult`；
- [ ] 响应由同一 DLL 释放；
- [ ] 先检查 C 返回码，再检查 JSON `ok`；
- [ ] 只采用 `aircraft_id + next_velocity_north_up_east_mps` 作为正式业务输出；
- [ ] 不重复施加诊断载荷；
- [ ] shutdown 前停止请求并等待全部线程；
- [ ] 保存模型哈希、Python/NumPy版本和部署包版本用于复现。

