# MNN - 轻量级深度学习推理引擎

[![License](https://img.shields.io/github/license/alibaba/MNN)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/Documentation-Read-green)](https://mnn-docs.readthedocs.io/en/latest/)
[![MNN Homepage](https://img.shields.io/badge/Homepage-Visit-green)](http://www.mnn.zone)

> 本项目 Fork 自 [alibaba/MNN](https://github.com/alibaba/MNN)，当前版本 **v3.4.1**

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
  - [PC/服务器构建](#pcserver-构建)
  - [Android 构建](#android-构建)
  - [iOS 构建](#ios-构建)
  - [鸿蒙构建](#鸿蒙构建)
  - [Python 安装](#python-安装)
- [模型推理](#模型推理)
  - [C++ 基础推理](#c-基础推理)
  - [Python 推理](#python-推理)
- [LLM 大语言模型](#llm-大语言模型)
  - [模型导出](#模型导出)
  - [LLM 推理](#llm-推理)
- [Diffusion 文生图](#diffusion-文生图)
- [模型转换](#模型转换)
- [硬件支持矩阵](#硬件支持矩阵)
- [配套工具](#配套工具)
- [应用程序](#应用程序)
- [文档与资源](#文档与资源)
- [常见问题](#常见问题)
- [许可证](#许可证)
- [更新日志](#更新日志)

---

## 项目简介

MNN（Mobile Neural Network）是阿里巴巴开发的高效轻量级深度学习推理框架，同时支持推理与训练。MNN 已在手机淘宝、天猫、优酷、钉钉、闲鱼等 30 多款 App 中使用，覆盖直播、短视频、搜索推荐、商品图搜、互动营销、安全风控等 70 多个场景。

MNN 包含以下核心子项目：
- **MNN 引擎**：深度学习推理核心，支持 CNN / RNN / GAN / Transformer 模型
- **MNN-LLM**：大语言模型端侧推理方案，支持 Qwen、LLAMA、Baichuan 等主流 LLM
- **MNN-Diffusion**：Stable Diffusion 端侧推理方案

---

## 核心特性

### 轻量级
- 无外部依赖，可部署到移动端和嵌入式设备
- Android 核心 so 体积约 **800KB**（armv7a, c++_shared）
- iOS 静态库约 **12MB**（armv7+arm64 全功能）
- 支持 `MNN_BUILD_MINI` 进一步减小约 25% 体积
- 支持 FP16 / Int8 模型量化，减小模型 50%~70%

### 高性能
- 大量手写汇编和 SIMD 优化（ARM NEON / x86 SSE/AVX/AVX512）
- GPU 推理：Metal (iOS/macOS) / OpenCL / Vulkan (Android) / CUDA (PC/Server)
- Winograd 卷积算法广泛应用
- ARMv8.2 FP16 半精度计算加速
- AVX512 / VNNI / BF16 等新架构指令支持

### 通用性
- 支持 TensorFlow / Caffe / ONNX / TorchScripts 模型格式
- 支持多输入多输出、动态输入、控制流模型
- 支持 178 个 TF OP、52 个 Caffe OP、163 个 TorchScripts OP、158 个 ONNX OP
- 支持 iOS 8.0+ / Android 4.3+ / Windows / macOS / Linux / 鸿蒙 / Web

### 易用性
- 类 NumPy 数值计算 API
- 类 OpenCV 的轻量图像处理库（MNN-CV，仅约 100KB）
- 支持 PC 和移动端模型训练
- 完善的 Python API

---

## 系统要求

| 平台 | 最低版本 | 编译工具 |
|------|---------|---------|
| Windows | Windows 10+ | Visual Studio 2019+, CMake 3.6+ |
| macOS | macOS 10.15+ | Xcode 12+, CMake 3.6+ |
| Linux | Ubuntu 18.04+ | GCC 7+, CMake 3.6+ |
| Android | Android 4.3+ (API 18) | Android NDK r21+ |
| iOS | iOS 8.0+ | Xcode 12+ |
| 鸿蒙 | OpenHarmony | DevEco Studio, OHOS SDK |

---

## 快速开始

### PC/Server 构建

#### 基础推理引擎

```bash
# Linux / macOS
mkdir build && cd build
cmake .. && make -j$(nproc)

# Windows (PowerShell)
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

#### 含 LLM 支持

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
```

#### 含测试和转换器

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_TEST=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON
make -j$(nproc)
```

#### 含 GPU 支持 (CUDA)

```bash
mkdir build && cd build
cmake .. -DMNN_CUDA=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
```

### Android 构建

使用统一构建脚本：

```bash
./build_lib.sh --android --ndk /path/to/android-ndk
```

或手动构建 arm64-v8a：

```bash
cd project/android
mkdir build_64 && cd build_64
cmake ../../../ \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DMNN_ARM82=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j4 MNN
```

### iOS 构建

```bash
# 真机 (arm64)
./build_lib.sh --ios

# 模拟器 (x86_64 + arm64)
./build_lib.sh --ios-simulator

# 两者都构建
./build_lib.sh --ios --ios-simulator
```

### 鸿蒙构建

```bash
./build_lib.sh --harmony --harmony-home /path/to/ohos/native
```

### Python 安装

#### 使用构建脚本

```bash
./build_lib.sh --python --python-deps llm
```

#### 使用 pip

```bash
cd pymnn/pip_package
python build_deps.py
python setup.py bdist_wheel
pip install dist/*.whl
```

---

## 模型推理

### C++ 基础推理

MNN 提供两套推理 API：

#### 方式一：Module API（推荐）

```cpp
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>

// 加载模型
auto module = Module::load({"input"}, {"output"}, "model.mnn");

// 准备输入
auto input = MNN::Express::_Input({1, 3, 224, 224}, NCHW);
auto inputPtr = input->writeMap<float>();
// ... 填充数据 ...

// 推理
auto output = module->onForward({input});
auto outputPtr = output[0]->readMap<float>();
// ... 使用结果 ...
```

#### 方式二：Session API（底层）

```cpp
#include <MNN/Interpreter.hpp>

// 创建解释器
auto interpreter = Interpreter::createFromFile("model.mnn");

// 创建会话
ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
auto session = interpreter->createSession(config);

// 获取输入张量并填充数据
auto inputTensor = interpreter->getSessionInput(session, nullptr);
// ... 填充数据 ...

// 运行推理
interpreter->runSession(session);

// 获取输出
auto outputTensor = interpreter->getSessionOutput(session, nullptr);
// ... 使用结果 ...

interpreter->releaseModel();
```

### Python 推理

```python
import MNN
import MNN.numpy as np
import MNN.cv as cv

# 使用 Module API
module = MNN.nn.load_module_from_file("model.mnn", ["input"], ["output"])

# 准备输入
input_var = MNN.expr.placeholder([1, 3, 224, 224], MNN.expr.NCHW)
# ... 填充数据 ...

# 推理
output = module.forward(input_var)
result = output.read()
```

---

## LLM 大语言模型

### 模型导出

1. **克隆 HuggingFace 模型**：

```bash
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git
```

2. **导出 MNN 模型**：

```bash
cd transformers/llm/export
python llmexport.py \
    --path /path/to/Qwen2-0.5B-Instruct \
    --export mnn
```

3. **导出文件说明**：

| 文件 | 用途 |
|------|------|
| `config.json` | 运行时配置 |
| `embeddings_bf16.bin` | Embedding 权重 |
| `llm.mnn` | MNN 模型文件 |
| `llm.mnn.weight` | MNN 模型权重 |
| `llm_config.json` | 模型配置 |
| `tokenizer.txt` | 分词器文件 |

**导出参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--path` | HuggingFace 模型路径 | 必填 |
| `--export` | 导出格式（`mnn` 或 `onnx`） | - |
| `--quant_bit` | 量化位数（4 或 8） | 4 |
| `--quant_block` | 量化块大小 | 0 (channel-wise) |
| `--lora_path` | LoRA 权重路径 | None |
| `--dst_path` | 输出路径 | `./model` |

### LLM 推理

#### 编译 LLM 推理引擎

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j$(nproc)
```

#### 运行推理

```bash
# 交互式对话
./llm_demo model_dir/config.json

# 批量处理 prompt 文件
./llm_demo model_dir/config.json prompt.txt

# 性能基准测试
./llm_bench -m model_dir/config.json
```

#### LLM 运行时配置 (config.json)

```json
{
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "max_new_tokens": 512
}
```

**关键配置项**：

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `backend_type` | 推理后端 (`cpu` / `opencl` / `metal`) | `cpu` |
| `thread_num` | CPU 线程数 | 4 |
| `precision` | 精度策略 (`low` / `normal` / `high`) | `low` (FP16) |
| `max_new_tokens` | 最大生成 token 数 | 512 |
| `use_mmap` | 使用 mmap 减少内存 | false |
| `attention_mode` | Attention 量化模式 (0-10) | 8 (Flash Attention) |
| `sampler_type` | 采样策略 (`greedy` / `temperature` / `mixed`) | `greedy` |

#### LoRA 支持

```bash
# 导出合并 LoRA 的模型
python llmexport.py --path /path/to/model --lora_path /path/to/lora --export mnn

# 导出分离 LoRA 的模型（支持运行时切换）
python llmexport.py --path /path/to/model --lora_path /path/to/lora --lora_split --export mnn
```

#### 视觉模型 (VL) 输入

在 prompt 中嵌入图片：
```
<img>https://example.com/image.jpg</img>描述这张图片的内容。
```

#### 音频模型输入

在 prompt 中嵌入音频：
```
<audio>https://example.com/audio.wav</audio>描述这段音频的内容。
```

---

## Diffusion 文生图

### 编译

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_OPENCV=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
```

详细使用说明请参考 [Diffusion README](./transformers/diffusion/README.md)。

---

## 模型转换

MNN 支持将以下格式的模型转换为 MNN 格式：

| 源格式 | 支持 OP 数量 |
|--------|-------------|
| TensorFlow | 178 |
| Caffe | 52 |
| TorchScripts | 163 |
| ONNX | 158 |

### 编译转换器

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j$(nproc)
```

### 转换示例

```bash
# ONNX 转 MNN
./MNNConvert --modelFile model.onnx --MNNModel model.mnn --framework ONNX

# TensorFlow 转 MNN
./MNNConvert --modelFile model.pb --MNNModel model.mnn --framework TF

# 带量化转换
./MNNConvert --modelFile model.onnx --MNNModel model.mnn -f ONNX \
    --weightQuantBits=4 --weightQuantBlock=128

# LLM 模型转换（含 Transformer 融合）
./MNNConvert --modelFile llm.onnx --MNNModel llm.mnn -f ONNX \
    --keepInputFormat --weightQuantBits=4 --weightQuantBlock=128 \
    --transformerFuse=1 --allowCustomOp --saveExternalData
```

---

## 硬件支持矩阵

- **S** ：深度优化，推荐使用
- **A** ：支持良好，可以使用
- **B** ：支持但未优化，不推荐
- **C** ：不支持

| 架构 / 精度 | | Normal | FP16 | BF16 | Int8 |
|---|---|---|---|---|---|
| **CPU** | Native | B | C | B | B |
| | x86/x64-SSE4.1 | A | C | C | A |
| | x86/x64-AVX2 | S | C | C | A |
| | x86/x64-AVX512 | S | C | C | S |
| | ARMv7a | S | S (ARMv8.2) | S | S |
| | ARMv8 | S | S (ARMv8.2) | S (ARMv8.6) | S |
| **GPU** | OpenCL | A | S | C | S |
| | Vulkan | A | A | C | A |
| | Metal | A | S | C | S |
| | CUDA | A | S | C | A |
| **NPU** | CoreML | A | C | C | C |
| | HIAI | A | C | C | C |
| | NNAPI | B | B | C | B |
| | QNN | C | B | C | C |

---

## 配套工具

| 工具 | 说明 |
|------|------|
| **MNN-Converter** | 模型格式转换（TF/Caffe/ONNX/TorchScripts → MNN），含图优化 |
| **MNN-Compress** | 模型压缩，在允许精度范围内减小体积、提升速度 |
| **MNN-Express** | 动态图计算接口，支持控制流和自定义计算 |
| **MNN-CV** | 轻量图像处理库（类 OpenCV，基于 MNN，仅约 100KB） |
| **MNN-Train** | 模型训练模块，支持 PC 和移动端训练 |

---

## 应用程序

| 应用 | 平台 | 说明 |
|------|------|------|
| [MNN Chat](./apps/Android/MnnLlmChat/README.md) | Android | 多模态 LLM 对话应用（文本/图像/音频/文生图） |
| [MNN TaoAvatar](./apps/Android/Mnn3dAvatar/README.md) | Android | 3D 数字人应用（LLM + ASR + TTS + 3D 渲染） |
| [MNN LLM iOS](./apps/iOS/MNNLLMChat/README.md) | iOS | 多模态 LLM 对话应用 |
| [Sana Image Edit](./apps/sana/README.md) | 跨平台 | 基于 Sana 的卡通风格照片编辑 |
| [MNN CLI](./apps/mnncli/) | 命令行 | 命令行推理工具 |

---

## 文档与资源

- **官方文档**：[https://mnn-docs.readthedocs.io](https://mnn-docs.readthedocs.io/en/latest/)
- **MNN 官网**：[http://www.mnn.zone](http://www.mnn.zone)
- **MNN Workbench**：可视化训练/部署工具，从官网下载
- **开发者代码文档**：[CODE_DOCUMENTATION.md](./CODE_DOCUMENTATION.md)
- **AI 开发指引**：[CLAUDE.md](./CLAUDE.md)
- **贡献指南**：[CONTRIBUTING.md](./CONTRIBUTING.md)

---

## 常见问题

### Q: 如何选择推理后端？

根据设备和场景选择：
- **CPU**：通用选择，兼容性最好
- **Metal**：iOS/macOS GPU 加速
- **OpenCL**：Android GPU 加速
- **CUDA**：PC/Server NVIDIA GPU 加速
- **Vulkan**：跨平台 GPU（Android/PC）

### Q: 如何减小库体积？

1. 使用 `MNN_BUILD_MINI=ON`（减小约 25%，但限制为固定输入大小）
2. 使用 `MNN_REDUCE_SIZE=ON`（移除不常用 OP）
3. 仅编译所需后端（不开启不需要的 GPU/NPU 选项）

### Q: LLM 推理内存不足怎么办？

1. 设置 `use_mmap: true`：将权重映射到磁盘
2. 设置 `kvcache_mmap: true`：将 KV Cache 映射到磁盘
3. 使用更低量化位数（4-bit）
4. 设置 `attention_mode: 10`：量化 QKV

### Q: Windows 构建报错？

- 确保使用 Visual Studio 2019+
- 使用 `MNN_WIN_RUNTIME_MT=ON` 解决运行时库冲突
- 转换器构建时 `MNN_BUILD_SHARED_LIBS=OFF` 需配合 `MNN_WIN_RUNTIME_MT=ON`

---

## 许可证

[Apache License 2.0](LICENSE.txt)

---

## 更新日志

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-04-05 | v3.4.1 | 基于 MNN v3.4.1 创建初始用户文档 |
| 2026/03/05 | - | 支持 Qwen3.5 系列模型 |
| 2026/02/13 | - | MNN-Sana-Edit-V2 卡通风格照片编辑 |
| 2025/10/16 | - | 支持 Qwen3-VL 系列 |
| 2025/06/11 | - | MNN TaoAvatar 3D 数字人应用发布 |
| 2025/05/12 | - | Android 支持 Qwen2.5 Omni 3B/7B |
| 2025/04/30 | - | Android 支持 Qwen3 + 暗黑模式 |
| 2025/02/18 | - | iOS 多模态 LLM App 发布 |
| 2025/02/11 | - | Android 支持 DeepSeek R1 1.5B |
| 2025/01/23 | - | Android 全模态 LLM App 发布 |

---

## 致谢

MNN 由阿里巴巴淘宝技术部、搜索工程团队、达摩院团队、优酷等团队开发。

在 MNN 的设计与开发中参考了以下开源项目：Caffe, FlatBuffers, gemmlowp, Halide, Mace, ONNX, Protobuf, skia, TensorFlow, ncnn, paddle-mobile, stb, rapidjson, pybind11, PyTorch, bolt, libyuv, libjpeg-turbo, OpenCV, ONNXRuntime。

## 引用

如果 MNN 对你的研究有所帮助，请引用以下论文：

```bibtex
@inproceedings{proc:osdi22:walle,
    author = {Chengfei Lv and Chaoyue Niu and others},
    title = {Walle: An End-to-End, General-Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning},
    booktitle = {OSDI 22},
    year = {2022},
}
```
