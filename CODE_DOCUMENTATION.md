# MNN 开发者代码文档 (CODE_DOCUMENTATION)

> 本文档面向开发者，用于理解 MNN 项目的代码架构、模块职责、构建方式和测试策略，以便后续进行功能开发和 Bug 修复。

## 1. 项目概述

MNN（Mobile Neural Network）是阿里巴巴开发的**高效轻量级深度学习推理引擎**，当前版本为 **3.4.1**。它支持深度学习模型的推理与训练，适用于服务器、个人电脑、手机、嵌入式设备和 IoT 设备。

MNN 的核心设计目标：
- **轻量级**：无外部依赖、极小的二进制体积（Android ARM so 约 800KB）
- **高性能**：大量手写汇编和 SIMD 优化，充分发挥硬件算力
- **跨平台**：支持 Windows / macOS / Linux / iOS / Android / 鸿蒙 / Web / 嵌入式
- **多后端**：CPU / GPU (Metal, OpenCL, Vulkan, CUDA) / NPU (CoreML, NNAPI, QNN)

---

## 2. 仓库目录结构

```
MNN/
├── include/MNN/             # 公开 C++ 头文件（用户级 API）
│   ├── MNNDefine.h          #   版本定义、日志宏、导出宏
│   ├── Interpreter.hpp      #   模型加载器（Session API 入口）
│   ├── Tensor.hpp           #   张量数据容器
│   ├── ImageProcess.hpp     #   图像预处理模块
│   ├── MNNForwardType.h     #   后端类型枚举、BackendConfig 定义
│   ├── ErrorCode.hpp        #   错误码定义
│   ├── Matrix.h             #   矩阵变换工具
│   ├── Rect.h               #   矩形区域工具
│   ├── AutoTime.hpp         #   计时工具
│   ├── MNNSharedContext.h   #   共享上下文（GPU 共享等）
│   └── expr/                #   Express API 头文件（高层动态图 API）
│       ├── Expr.hpp         #     表达式节点（VARP 变量）
│       ├── Module.hpp       #     模型加载与前向推理（推荐 API）
│       ├── Executor.hpp     #     执行器管理
│       ├── ExecutorScope.hpp#     执行器作用域
│       ├── MathOp.hpp       #     数学运算 OP
│       ├── NeuralNetWorkOp.hpp #  神经网络 OP
│       ├── Optimizer.hpp    #     图优化器
│       └── Scope.hpp        #     作用域管理
│
├── source/                  # 核心源代码
│   ├── core/                #   推理引擎核心
│   │   ├── Interpreter.cpp  #     模型解析与加载
│   │   ├── Session.cpp/.hpp #     推理会话管理
│   │   ├── Pipeline.cpp/.hpp#     算子流水线调度
│   │   ├── Schedule.cpp/.hpp#     图调度与任务分配
│   │   ├── Backend.cpp/.hpp #     后端抽象基类
│   │   ├── Execution.cpp/.hpp#    算子执行抽象基类
│   │   ├── Tensor.cpp       #     张量实现
│   │   ├── TensorUtils.cpp/.hpp#  张量工具（格式转换等）
│   │   ├── KVCacheManager.cpp/.hpp # KV 缓存管理（LLM 用）
│   │   ├── BufferAllocator.cpp/.hpp# 内存分配器
│   │   ├── ConvolutionCommon.cpp/.hpp# 卷积公共逻辑
│   │   ├── OpCommonUtils.cpp/.hpp#  算子公共工具
│   │   ├── FileLoader.cpp/.hpp#    文件加载器
│   │   ├── WrapExecution.cpp/.hpp# 执行包装（精度转换等）
│   │   └── ...
│   │
│   ├── backend/             #   硬件后端实现
│   │   ├── cpu/             #     CPU 后端（含大量 SIMD/汇编优化）
│   │   ├── arm82/           #     ARMv8.2 FP16 加速后端
│   │   ├── metal/           #     Apple Metal GPU 后端
│   │   ├── cuda/            #     NVIDIA CUDA GPU 后端
│   │   ├── opencl/          #     OpenCL GPU 后端
│   │   ├── vulkan/          #     Vulkan GPU 后端
│   │   ├── opengl/          #     OpenGL GPU 后端
│   │   ├── musa/            #     摩尔线程 MUSA GPU 后端
│   │   ├── coreml/          #     Apple CoreML NPU 后端
│   │   ├── nnapi/           #     Android NNAPI NPU 后端
│   │   ├── qnn/             #     Qualcomm QNN 后端
│   │   ├── hiai/            #     华为 HiAI 后端
│   │   ├── tensorrt/        #     NVIDIA TensorRT 后端
│   │   └── neuropilot/      #     联发科 NeuroPilot 后端
│   │
│   ├── shape/               #   形状推理（Shape Inference）
│   ├── geometry/            #   几何计算（OP 分解/融合）
│   ├── math/                #   数学运算内核
│   ├── cv/                  #   计算机视觉基础运算
│   ├── utils/               #   内部工具函数
│   ├── jni/                 #   JNI 接口（Java/Android 绑定）
│   └── plugin/              #   插件系统
│
├── express/                 # Express API 实现（高层动态图）
│   ├── Executor.cpp         #   执行器实现
│   ├── Expr.cpp             #   表达式节点实现
│   ├── MathOp.cpp           #   数学运算 OP 实现
│   ├── NeuralNetWorkOp.cpp  #   神经网络 OP 实现
│   ├── Utils.cpp/.hpp       #   Express 工具函数
│   └── module/              #   Module（模型）实现
│
├── schema/                  # FlatBuffers 模式定义
│   ├── default/             #   OP 定义（.fbs 文件）
│   └── current/             #   当前生成的头文件
│
├── tools/                   # 工具集
│   ├── converter/           #   模型转换器（ONNX/TF/Caffe/Torchscripts → MNN）
│   ├── train/               #   训练框架
│   ├── quantization/        #   量化工具
│   ├── cv/                  #   MNN-CV（轻量级 OpenCV 替代）
│   ├── audio/               #   音频处理
│   ├── cpp/                 #   C++ 工具程序
│   ├── mnncompress/         #   模型压缩
│   ├── evaluation/          #   评估工具
│   └── script/              #   测试辅助脚本（modelTest.py 等）
│
├── transformers/            # Transformer 模型支持
│   ├── llm/                 #   LLM（大语言模型）推理引擎
│   │   ├── export/          #     模型导出（Python: HuggingFace → MNN）
│   │   │   ├── llmexport.py #       LLM 导出入口
│   │   │   ├── gguf2mnn.py  #       GGUF 格式转 MNN
│   │   │   ├── safetensors2mnn.py # SafeTensors 转 MNN
│   │   │   └── utils/       #       导出工具（model_mapper, transformers 等）
│   │   ├── engine/          #     LLM C++ 推理引擎
│   │   │   ├── src/         #       核心推理实现（llm.cpp, omni.cpp 等）
│   │   │   ├── include/     #       公开头文件（llm.hpp, reranker.hpp）
│   │   │   ├── demo/        #       演示程序
│   │   │   ├── app/         #       应用程序
│   │   │   ├── test/        #       LLM 测试
│   │   │   ├── tools/       #       LLM 工具
│   │   │   └── ios/         #       iOS 集成
│   │   ├── benchmark/       #     性能基准测试
│   │   └── eval/            #     评估工具
│   │
│   └── diffusion/           #   Stable Diffusion 支持
│       ├── export/          #     Diffusion 模型导出
│       └── engine/          #     Diffusion C++ 推理引擎
│
├── apps/                    # 应用程序
│   ├── Android/             #   Android App（MNN Chat, 3D Avatar 等）
│   ├── iOS/                 #   iOS App（MNN LLM Chat）
│   ├── mnncli/              #   命令行工具
│   ├── sana/                #   Sana 图像编辑
│   └── frameworks/          #   框架集成
│
├── pymnn/                   # Python 绑定
│   ├── src/                 #   C++ 绑定源码
│   ├── pip_package/         #   pip 包构建
│   ├── examples/            #   Python 使用示例
│   └── test/                #   Python 测试
│
├── test/                    # C++ 测试套件
│   ├── core/                #   核心功能测试
│   ├── op/                  #   算子测试
│   ├── backend/             #   后端测试
│   ├── cv/                  #   CV 测试
│   ├── expr/                #   Express 测试
│   ├── grad/                #   梯度测试
│   ├── model/               #   模型测试
│   ├── speed/               #   性能测试
│   └── MNNTestSuite.h/.cpp  #   测试框架
│
├── benchmark/               # 性能基准测试
├── demo/                    # 演示程序
├── 3rd_party/               # 第三方依赖（flatbuffers, protobuf, half 等）
├── cmake/                   # CMake 模块和工具链
├── project/                 # 平台特定构建项目
├── skills/                  # AI 代理 Skill 指令
│   ├── support-new-llm/     #   添加新 LLM 支持
│   ├── add-new-op/          #   添加新算子
│   └── arm-cpu-optimize/    #   ARM CPU 优化
│
├── CMakeLists.txt           # 主 CMake 构建文件
├── build_lib.sh             # 统一构建脚本（Android/iOS/鸿蒙/Python）
├── test.sh                  # Linux/macOS 测试脚本
├── test.ps1                 # Windows 测试脚本
├── test.bat                 # Windows 批处理测试脚本
├── CLAUDE.md                # AI 代理指令文件
└── CONTRIBUTING.md          # 贡献指南
```

---

## 3. 核心架构

### 3.1 推理架构：图优化 + 异构后端调度

MNN 的推理流程遵循以下管线：

```
模型文件 (.mnn)
    ↓
Interpreter（模型解析）
    ↓
Schedule（图调度：后端选择 + 任务分配）
    ↓
Pipeline（流水线：OP 逐个执行）
    ↓
Backend + Execution（硬件后端 + 算子实现）
    ↓
输出 Tensor
```

### 3.2 两套推理 API

| API | 层次 | 入口 | 适用场景 |
|-----|------|------|---------|
| **Session API**（低层） | 底层直接操作 | `Interpreter → createSession → runSession` | 精确控制内存、旧代码兼容 |
| **Module API**（高层，推荐） | 基于 Express 动态图 | `Module::load → onForward(VARP)` | LLM / Diffusion / 现代工作负载 |

### 3.3 关键抽象

| 抽象 | 头文件位置 | 职责 |
|------|-----------|------|
| `Interpreter` | `include/MNN/Interpreter.hpp` | 模型文件解析、Session 管理 |
| `Session` | `source/core/Session.hpp` | 推理会话，管理 Pipeline 和内存 |
| `Pipeline` | `source/core/Pipeline.hpp` | 算子执行流水线 |
| `Schedule` | `source/core/Schedule.hpp` | 图调度：后端选择与任务分配 |
| `Backend` | `source/core/Backend.hpp` | 硬件后端抽象基类 |
| `Execution` | `source/core/Execution.hpp` | 单个算子的执行抽象基类 |
| `Tensor` | `include/MNN/Tensor.hpp` | 数据容器，内部使用 NC4HW4 格式 |
| `Module` | `include/MNN/expr/Module.hpp` | 高层模型封装（推荐使用） |
| `Executor` | `include/MNN/expr/Executor.hpp` | Express 执行器管理 |
| `VARP` | `include/MNN/expr/Expr.hpp` | Express 变量（动态图节点） |

### 3.4 算子注册模式

添加新算子的标准流程：

```
Schema 定义 (.fbs)
    ↓
Shape 推理 (source/shape/)
    ↓
Geometry 分解 (source/geometry/) [可选]
    ↓
Backend Execution 实现 (source/backend/<后端名>/)
```

### 3.5 Tensor 数据格式

MNN 内部默认使用 **NC4HW4** 格式（通道按 4 对齐打包），这是为 SIMD 指令优化的核心格式。用户输入输出通常使用 NCHW 或 NHWC 格式，框架会自动转换。

---

## 4. 构建系统

### 4.1 构建工具

- **构建系统**：CMake（最低版本 3.6）
- **C 标准**：C99
- **C++ 标准**：默认 C++11，CUDA + Transformer Fuse 时使用 C++17
- **编译器限制**：禁用 RTTI（`-fno-rtti`）和异常（`-fno-exceptions`）

### 4.2 核心 CMake 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `MNN_BUILD_SHARED_LIBS` | ON | 构建动态库或静态库 |
| `MNN_BUILD_TEST` | OFF | 构建测试 |
| `MNN_BUILD_CONVERTER` | OFF | 构建模型转换器 |
| `MNN_BUILD_QUANTOOLS` | OFF | 构建量化工具 |
| `MNN_BUILD_TRAIN` | OFF | 构建训练框架 |
| `MNN_BUILD_DEMO` | OFF | 构建演示程序 |
| `MNN_BUILD_TOOLS` | ON | 构建 C++ 工具 |
| `MNN_BUILD_LLM` | OFF | 构建 LLM 推理库 |
| `MNN_BUILD_LLM_OMNI` | OFF | 构建 LLM 多模态（视觉/音频） |
| `MNN_BUILD_DIFFUSION` | OFF | 构建 Diffusion 推理 |
| `MNN_BUILD_OPENCV` | OFF | 构建 MNN-CV（OpenCV 替代） |
| `MNN_BUILD_AUDIO` | OFF | 构建音频处理 |
| `MNN_BUILD_MINI` | OFF | 最小化构建（固定输入大小） |
| `MNN_LOW_MEMORY` | OFF | 低内存模式（权重量化模型支持） |
| `MNN_SUPPORT_TRANSFORMER_FUSE` | OFF | Transformer 融合算子支持 |
| `MNN_SEP_BUILD` | ON | 后端和 Express 分离构建 |
| `MNN_REDUCE_SIZE` | OFF | 精简包大小（移除不常用 OP） |

### 4.3 后端相关选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `MNN_METAL` | OFF | Apple Metal GPU |
| `MNN_OPENCL` | OFF | OpenCL GPU |
| `MNN_VULKAN` | OFF | Vulkan GPU |
| `MNN_CUDA` | OFF | NVIDIA CUDA GPU |
| `MNN_MUSA` | OFF | 摩尔线程 MUSA GPU |
| `MNN_ARM82` | ON | ARMv8.2 FP16 加速 |
| `MNN_AVX2` | ON | x86 AVX2 加速 |
| `MNN_AVX512` | OFF | x86 AVX512 加速 |
| `MNN_COREML` | OFF | Apple CoreML NPU |
| `MNN_NNAPI` | OFF | Android NNAPI NPU |
| `MNN_QNN` | OFF | Qualcomm QNN |
| `MNN_TENSORRT` | OFF | NVIDIA TensorRT |
| `MNN_ONEDNN` | OFF | Intel oneDNN |

### 4.4 构建命令示例

#### PC / 服务器（基础推理）

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

#### PC / 服务器（含 LLM 支持）

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)
```

#### Windows (MSVC)

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
cmake --build . --config Release
```

#### 含转换器和量化工具

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON && make -j$(nproc)
```

#### 带 GPU 支持（CUDA）

```bash
mkdir build && cd build
cmake .. -DMNN_CUDA=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)
```

#### 跨平台统一构建脚本

```bash
# Android
./build_lib.sh --android --ndk /path/to/ndk

# iOS 真机 + 模拟器
./build_lib.sh --ios --ios-simulator

# 鸿蒙
./build_lib.sh --harmony --harmony-home /path/to/ohos/native

# Python
./build_lib.sh --python --python-deps llm,opencl

# 全平台
./build_lib.sh --android --ios --harmony --python --ndk /path/to/ndk
```

---

## 5. LLM 子系统

### 5.1 架构

LLM 子系统分为两部分：

1. **模型导出**（Python，`transformers/llm/export/`）：将 HuggingFace 模型转换为 MNN 格式
2. **模型推理**（C++，`transformers/llm/engine/`）：在设备端加载和运行 LLM 模型

### 5.2 导出流程

```
HuggingFace 模型
    ↓ llmexport.py
ONNX 模型 + Tokenizer + Embedding
    ↓ MNNConvert（内部调用或手动）
MNN 模型文件 (.mnn + .mnn.weight)
```

导出关键文件：
- `llmexport.py`：导出入口脚本
- `utils/model_mapper.py`：模型字段映射
- `utils/model.py`：统一 LlmModel 类
- `utils/transformers.py`：Attention/Decoder/RoPE 导出

### 5.3 推理引擎

- `llm.cpp`：文本推理核心
- `omni.cpp`：多模态推理（视觉 + 音频）
- 包含 KVCache 管理、采样策略（greedy, temperature, topK, topP 等）

### 5.4 LLM 配置 (config.json)

关键配置项：
- **模型文件**：`llm_model`, `llm_weight`, `embedding_file`, `tokenizer_file`
- **推理配置**：`max_new_tokens`, `attention_mode`, `use_mmap`, `kvcache_mmap`
- **硬件配置**：`backend_type` (cpu/opencl/metal), `thread_num`, `precision`
- **采样配置**：`sampler_type`, `temperature`, `topK`, `topP`, `penalty`

---

## 6. Diffusion 子系统

路径：`transformers/diffusion/`

- `export/`：Stable Diffusion 模型导出
- `engine/`：C++ 推理引擎
- 支持 Sana 等文生图/图像编辑模型

---

## 7. 代码风格

### 7.1 C++ 风格

- **风格规范**：Google Style 变体（参见 `.clang-format`）
- **缩进**：4 空格
- **行宽**：120 字符
- **大括号**：附加式（attached braces）
- **命名规则**：
  - 类名：`PascalCase`（如 `ConvolutionCommon`）
  - 函数名：`camelCase`（如 `createSession`）
  - 成员变量：`mCamelCase`（如 `mSession`）
- **编译限制**：禁用 RTTI 和异常
- **格式化命令**：`clang-format -i -style=file <file>`

### 7.2 Python 风格

- 标准 Python 约定

---

## 8. 测试策略

### 8.1 测试类型

| 测试类型 | 位置 | 说明 |
|---------|------|------|
| 单元测试 | `test/` | 算子、核心功能、Express API 测试 |
| 模型测试 | `test/model/` | 端到端模型推理测试 |
| 后端测试 | `test/backend/` | 特定后端正确性测试 |
| CV 测试 | `test/cv/` | 计算机视觉操作测试 |
| 梯度测试 | `test/grad/` | 自动微分/梯度测试 |
| 性能测试 | `test/speed/` | 性能基准测试 |
| LLM 测试 | `transformers/llm/engine/test/` | LLM 推理测试 |
| Python 测试 | `pymnn/test/` | Python 绑定测试 |
| 转换测试 | 通过 `test.sh`/`test.ps1` | ONNX/TF/TFLite/Torch 转换测试 |

### 8.2 运行测试

```bash
# 构建测试
mkdir build && cd build
cmake .. -DMNN_BUILD_TEST=ON && make -j$(nproc)

# 运行单元测试
./run_test.out

# 运行特定类型测试（op 测试 + 多线程）
./run_test.out op 0 0 4

# LLM 推理测试
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM 性能基准
./llm_bench -m /path/to/MODEL/config.json

# Windows 测试
.\test.ps1         # 基本测试
.\test.ps1 -gpu    # 含 GPU 测试
.\test.ps1 -x86    # 32 位测试
```

### 8.3 测试框架

MNN 使用自定义测试框架 `MNNTestSuite`（`test/MNNTestSuite.h`），类似于 Google Test 的注册机制。

---

## 9. CI/CD 工作流

### 9.1 GitHub Actions 工作流一览

| 工作流文件 | 名称 | 用途 |
|-----------|------|------|
| `android.yml` | android | 编译 MNN 原生库（arm64 + arm32），验证 C++ 编译 |
| `android-apk.yml` | android-apk | 完整构建 MnnLlmChat standardDebug APK（NDK 编译 + Gradle 构建） |
| `linux.yml` | linux | Linux 平台编译测试 |
| `macos.yml` | macos | macOS 平台编译测试 |
| `windows.yml` | windows | Windows 平台编译测试 |
| `ios.yml` | ios | iOS 平台编译测试 |
| `code-format.yml` | code-format | 代码格式检查 |

### 9.2 android-apk 工作流详解

**文件**: `.github/workflows/android-apk.yml`

**触发条件**: 所有分支 push、PR 到 master、手动触发

**构建流程**:
1. 安装 JDK 17，复用 runner 预装的 Android SDK，仅额外安装 NDK 27.2.12479018
2. CMake 编译 MNN 原生库（arm64-v8a），参数复刻自 `apps/Android/MnnLlmChat/build.sh`
3. Gradle 构建 `assembleStandardDebug`（构建前 unset `ANDROID_SDK_ROOT` 以避免与 `ANDROID_HOME` 冲突）
4. 上传 APK 为 GitHub Actions Artifact

**关键依赖**:
- `project/android/build_64.sh`: NDK 编译脚本，依赖 `$ANDROID_NDK` 环境变量
- `apps/Android/MnnLlmChat/`: Gradle 项目，依赖 `project/android/build_64/lib/` 下的 `.so` 文件
- NDK 版本必须精确匹配 `27.2.12479018`（`app/build.gradle` 硬编码）
- `ANDROID_HOME` 使用 runner 预装值，不自定义（否则与 `ANDROID_SDK_ROOT` 冲突）

---

## 10. Skills（AI 辅助开发指引）

`skills/` 目录包含面向 AI 代理的结构化开发指引：

| Skill | 入口文件 | 使用场景 |
|-------|---------|---------|
| 支持新 LLM | `skills/support-new-llm/SKILL.md` | 添加或适配新的大语言模型 |
| 添加新算子 | `skills/add-new-op/SKILL.md` | 添加新的计算算子 |
| ARM CPU 优化 | `skills/arm-cpu-optimize/SKILL.md` | 优化 ARM CPU 上的算子性能 |

使用前**必须先阅读对应的 SKILL.md 文件**，然后按步骤执行。

---

## 11. 限制区域

> ⚠️ 以下目录包含内部私有代码，**禁止读取、修改或引用**：
> - `schema/private/`
> - `source/internal/`

---

## 12. 依赖关系

### 12.1 内部模块依赖（构建顺序）

```
MNNCore (核心)
    ↓
MNNMath (数学库) + MNNCV (CV 库)
    ↓
MNNUtils (工具)
    ↓
MNNTransform (形状推理 + 几何计算) [可选，MNN_SKIPBUILD_GEOMETRY=OFF 时]
    ↓
MNNCPU (CPU 后端)
    ↓
[可选后端: Metal, CUDA, OpenCL, Vulkan, ...]
    ↓
MNN_Express (Express API) [可选，MNN_SKIPBUILD_GEOMETRY=OFF 时]
    ↓
[可选模块: Train, CV, Audio, LLM, Diffusion, Converter, ...]
```

### 12.2 第三方依赖

| 依赖 | 路径 | 用途 |
|------|------|------|
| FlatBuffers | `3rd_party/flatbuffers/` | 模型序列化格式 |
| Protobuf | `3rd_party/protobuf/` | 转换器（TF 模型解析） |
| half | `3rd_party/half/` | 半精度浮点支持 |
| imageHelper | `3rd_party/imageHelper/` | 图像加载辅助 |
| OpenCLHeaders | `3rd_party/OpenCLHeaders/` | OpenCL 头文件 |

### 12.3 LLM 导出依赖（Python）

见 `transformers/llm/export/requirements.txt`

---

## 13. 更新日志

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-04-05 | 初始版本 | 基于 MNN v3.4.1 源码创建初始代码文档 |
| 2026-04-05 | CI 更新 | 新增 `android-apk.yml` 工作流及第 9 章 CI/CD 文档 |
| 2026-04-05 | CI 修复 | 修复 `android-apk.yml` SDK 路径冲突：复用 runner 预装 SDK，不再自建 SDK 目录 |
