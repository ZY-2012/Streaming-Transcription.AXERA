#!/usr/bin/env bash
# 下载交叉编译器、AX650 BSP SDK 与 ALSA 头文件（默认放到 toolchains/，不随 git 提交）。
#
# 用法:
#   bash scripts/download_toolchains.sh
#
# 若已把工具链放在其他目录，可导出环境变量后直接编译：
#   export CXX=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
#   export AX650_SDK_ROOT=/path/to/ax650n_bsp_sdk/msp/out
#   export ALSA_INCLUDE_DIR=/path/to/include     # 含 alsa/ 子目录
#   export ALSA_LIBRARY=/path/to/libasound.so    # ARM64 版本
#   bash scripts/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAINS_DIR="${ROOT}/toolchains"

GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
GCC_DIR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

# 1. gcc 交叉编译器
if [[ -x "${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/${GCC_DIR}"
else
    mkdir -p "${TOOLCHAINS_DIR}"
    cd "${TOOLCHAINS_DIR}"
    echo "Downloading ${GCC_URL}"
    wget -q --show-progress "${GCC_URL}" -O gcc.tar.xz
    tar -xf gcc.tar.xz
    rm -f gcc.tar.xz
    echo "Done: ${TOOLCHAINS_DIR}/${GCC_DIR}"
fi

# 2. AX650 BSP SDK（ax_engine/ax_sys 头文件与库）
if [[ -f "${TOOLCHAINS_DIR}/ax650n_bsp_sdk/msp/out/include/ax_engine_api.h" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
else
    mkdir -p "${TOOLCHAINS_DIR}"
    cd "${TOOLCHAINS_DIR}"
    echo "Cloning https://github.com/AXERA-TECH/ax650n_bsp_sdk.git"
    git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
    echo "Done: ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
fi

# 3. ALSA 头文件（来自 alsa-lib 官方源码包；仅编译 run-alsa-apm 需要）
if [[ -f "${TOOLCHAINS_DIR}/alsa/include/alsa/asoundlib.h" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/alsa"
else
    mkdir -p "${TOOLCHAINS_DIR}"
    cd "${TOOLCHAINS_DIR}"
    ALSA_VER="1.2.8"
    ALSA_URL="https://www.alsa-project.org/files/pub/lib/alsa-lib-${ALSA_VER}.tar.bz2"
    echo "Downloading ${ALSA_URL}"
    wget -q --show-progress "${ALSA_URL}" -O alsa-lib.tar.bz2
    tar -xf alsa-lib.tar.bz2
    # 只取头文件：include/alsa 即官方安装布局
    mkdir -p alsa
    cp -r "alsa-lib-${ALSA_VER}/include/alsa" alsa/
    rm -rf "alsa-lib-${ALSA_VER}" alsa-lib.tar.bz2
    echo "Done: ${TOOLCHAINS_DIR}/alsa"
fi

echo ""
echo "工具链就绪。编译示例："
echo "  export CXX=${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++"
echo "  export AX650_SDK_ROOT=${TOOLCHAINS_DIR}/ax650n_bsp_sdk/msp/out"
echo "  export ALSA_INCLUDE_DIR=${TOOLCHAINS_DIR}/alsa"
echo "  bash scripts/build.sh"
