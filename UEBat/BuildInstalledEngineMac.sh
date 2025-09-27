#!/bin/bash

# ===============================================
# UE5.6 Mac 安装构建脚本
# 专门用于Mac平台的安装构建，包含Mac server编译
# ===============================================

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "用法: BuildInstalledEngineMac.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --help, -h                    显示此帮助信息"
    echo "  --build-dir DIR               指定构建输出目录 (默认: ./LocalBuilds/Engine)"
    echo "  --with-server                 包含Mac server目标 (默认: true)"
    echo "  --with-client                 包含Mac client目标 (默认: false)"
    echo "  --with-editor                 包含Mac editor (默认: true)"
    echo "  --with-ddc                    构建DDC (默认: true)"
    echo "  --configurations CONFIGS      指定游戏配置 (默认: Shipping;Development;DebugGame)"
    echo "  --sign-executables            签名可执行文件 (默认: false)"
    echo "  --full-debug-info             生成完整调试信息 (默认: false)"
    echo "  --extra-compile-args ARGS     额外的编译参数"
    echo "  --build-id ID                 构建标识符"
    echo "  --verbose                     详细输出"
    echo "  --clean                       清理构建"
    echo "  --no-parallel                 禁用并行构建"
    echo "  --max-jobs N                  最大并行作业数 (默认: 4)"
    echo ""
    echo "示例:"
    echo "  BuildInstalledEngineMac.sh --with-server --verbose"
    echo "  BuildInstalledEngineMac.sh --build-dir \"/Users/MyBuilds/UE5\" --with-client --with-server"
    echo "  BuildInstalledEngineMac.sh --configurations \"Shipping;Development\" --sign-executables"
    echo ""
}

# 检查是否在正确的目录中运行
check_environment() {
    if [ ! -f "Engine/Build/BatchFiles/Build.sh" ]; then
        log_error "此脚本必须在UE5.6引擎根目录中运行"
        log_error "请确保脚本位于包含Engine文件夹的目录中"
        exit 1
    fi
    
    # 检查是否为Mac系统
    if [ "$(uname)" != "Darwin" ]; then
        log_error "此脚本只能在Mac系统上运行"
        exit 1
    fi
    
    log_success "环境检查通过"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Xcode
    if ! command -v xcodebuild &> /dev/null; then
        log_error "未找到Xcode，请安装Xcode Command Line Tools"
        log_info "运行: xcode-select --install"
        exit 1
    fi
    
    # 检查.NET
    if ! command -v dotnet &> /dev/null; then
        log_error "未找到.NET SDK，请安装.NET 6.0或更高版本"
        exit 1
    fi
    
    # 检查dotnet版本
    DOTNET_VERSION=$(dotnet --version | cut -d. -f1)
    if [ "$DOTNET_VERSION" -lt 6 ]; then
        log_error "需要.NET 6.0或更高版本，当前版本: $(dotnet --version)"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 设置默认值
BUILD_DIR="./LocalBuilds/Engine"
WITH_SERVER=true
WITH_CLIENT=false
WITH_EDITOR=true
WITH_DDC=true
GAME_CONFIGURATIONS="Shipping;Development;DebugGame"
SIGN_EXECUTABLES=false
WITH_FULL_DEBUG_INFO=false
EXTRA_COMPILE_ARGS=""
BUILD_ID=""
VERBOSE=false
CLEAN_BUILD=false
PARALLEL_BUILD=true
MAX_PARALLEL_JOBS=4
LOG_FILE=""

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            --with-server)
                WITH_SERVER=true
                shift
                ;;
            --with-client)
                WITH_CLIENT=true
                shift
                ;;
            --with-editor)
                WITH_EDITOR=true
                shift
                ;;
            --no-editor)
                WITH_EDITOR=false
                shift
                ;;
            --with-ddc)
                WITH_DDC=true
                shift
                ;;
            --no-ddc)
                WITH_DDC=false
                shift
                ;;
            --configurations)
                GAME_CONFIGURATIONS="$2"
                shift 2
                ;;
            --sign-executables)
                SIGN_EXECUTABLES=true
                shift
                ;;
            --full-debug-info)
                WITH_FULL_DEBUG_INFO=true
                shift
                ;;
            --extra-compile-args)
                EXTRA_COMPILE_ARGS="$2"
                shift 2
                ;;
            --build-id)
                BUILD_ID="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --no-parallel)
                PARALLEL_BUILD=false
                shift
                ;;
            --max-jobs)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 显示构建配置
show_config() {
    log_info "构建配置:"
    echo "  构建目录: $BUILD_DIR"
    echo "  Mac Server: $WITH_SERVER"
    echo "  Mac Client: $WITH_CLIENT"
    echo "  Mac Editor: $WITH_EDITOR"
    echo "  DDC构建: $WITH_DDC"
    echo "  游戏配置: $GAME_CONFIGURATIONS"
    echo "  签名可执行文件: $SIGN_EXECUTABLES"
    echo "  完整调试信息: $WITH_FULL_DEBUG_INFO"
    echo "  详细输出: $VERBOSE"
    echo "  清理构建: $CLEAN_BUILD"
    echo "  并行构建: $PARALLEL_BUILD"
    echo "  最大并行作业: $MAX_PARALLEL_JOBS"
    echo ""
}

# 创建构建目录
create_build_directory() {
    log_info "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

# 清理构建
clean_build() {
    if [ "$CLEAN_BUILD" = true ]; then
        log_info "清理构建目录..."
        if [ -d "$BUILD_DIR" ]; then
            rm -rf "$BUILD_DIR"
            mkdir -p "$BUILD_DIR"
        fi
    fi
}

# 设置日志文件
setup_logging() {
    if [ -z "$LOG_FILE" ]; then
        LOG_FILE="./BuildLog_$(date +%Y%m%d_%H%M%S).log"
    fi
    
    log_info "日志文件: $LOG_FILE"
}

# 构建UAT和UBT
build_tools() {
    log_info "构建构建工具..."
    
    # 构建UAT
    log_info "构建UAT..."
    if [ "$VERBOSE" = true ]; then
        ./Engine/Build/BatchFiles/BuildUAT.sh
    else
        ./Engine/Build/BatchFiles/BuildUAT.sh >> "$LOG_FILE" 2>&1
    fi
    
    if [ $? -ne 0 ]; then
        log_error "UAT构建失败"
        exit 1
    fi
    
    # 构建UBT
    log_info "构建UBT..."
    if [ "$VERBOSE" = true ]; then
        ./Engine/Build/BatchFiles/BuildUBT.sh
    else
        ./Engine/Build/BatchFiles/BuildUBT.sh >> "$LOG_FILE" 2>&1
    fi
    
    if [ $? -ne 0 ]; then
        log_error "UBT构建失败"
        exit 1
    fi
    
    log_success "构建工具构建完成"
}

# 准备UAT命令
prepare_uat_command() {
    UAT_COMMAND="BuildInstalledEngine"
    UAT_ARGS=""
    
    # 添加基本参数
    UAT_ARGS="$UAT_ARGS -BuiltDirectory=$BUILD_DIR"
    
    # 添加平台参数 - 仅Mac
    UAT_ARGS="$UAT_ARGS -WithMac=true"
    UAT_ARGS="$UAT_ARGS -WithWin64=false"
    UAT_ARGS="$UAT_ARGS -WithLinux=false"
    UAT_ARGS="$UAT_ARGS -WithAndroid=false"
    UAT_ARGS="$UAT_ARGS -WithIOS=false"
    UAT_ARGS="$UAT_ARGS -WithTVOS=false"
    UAT_ARGS="$UAT_ARGS -WithLinuxArm64=false"
    
    # 添加目标类型参数
    if [ "$WITH_CLIENT" = true ]; then
        UAT_ARGS="$UAT_ARGS -WithClient=true"
    else
        UAT_ARGS="$UAT_ARGS -WithClient=false"
    fi
    
    if [ "$WITH_SERVER" = true ]; then
        UAT_ARGS="$UAT_ARGS -WithServer=true"
    else
        UAT_ARGS="$UAT_ARGS -WithServer=false"
    fi
    
    # 添加编辑器参数
    if [ "$WITH_EDITOR" = true ]; then
        UAT_ARGS="$UAT_ARGS -WithInstalledMac=true"
    else
        UAT_ARGS="$UAT_ARGS -WithInstalledMac=false"
    fi
    
    # 添加DDC参数
    if [ "$WITH_DDC" = true ]; then
        UAT_ARGS="$UAT_ARGS -WithDDC=true"
        UAT_ARGS="$UAT_ARGS -HostPlatformDDCOnly=true"
    else
        UAT_ARGS="$UAT_ARGS -WithDDC=false"
    fi
    
    # 添加其他参数
    if [ "$SIGN_EXECUTABLES" = true ]; then
        UAT_ARGS="$UAT_ARGS -SignExecutables=true"
    fi
    
    if [ -n "$GAME_CONFIGURATIONS" ]; then
        UAT_ARGS="$UAT_ARGS -GameConfigurations=$GAME_CONFIGURATIONS"
    fi
    
    if [ "$WITH_FULL_DEBUG_INFO" = true ]; then
        UAT_ARGS="$UAT_ARGS -WithFullDebugInfo=true"
    fi
    
    if [ -n "$EXTRA_COMPILE_ARGS" ]; then
        UAT_ARGS="$UAT_ARGS -ExtraCompileArgs=$EXTRA_COMPILE_ARGS"
    fi
    
    if [ -n "$BUILD_ID" ]; then
        UAT_ARGS="$UAT_ARGS -BuildIdOverride=$BUILD_ID"
    fi
    
    # 添加并行构建参数
    if [ "$PARALLEL_BUILD" = true ]; then
        UAT_ARGS="$UAT_ARGS -MaxParallelJobs=$MAX_PARALLEL_JOBS"
    fi
    
    # 添加详细输出参数
    if [ "$VERBOSE" = true ]; then
        UAT_ARGS="$UAT_ARGS -msbuild-verbose"
    fi
}

# 运行构建
run_build() {
    log_info "开始构建安装引擎..."
    log_info "执行命令: ./Engine/Build/BatchFiles/RunUAT.sh $UAT_COMMAND $UAT_ARGS"
    
    # 记录构建开始时间
    echo "构建开始时间: $(date)" >> "$LOG_FILE"
    echo "构建配置: Mac Server Build" >> "$LOG_FILE"
    echo "构建目录: $BUILD_DIR" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # 运行UAT构建
    if [ "$VERBOSE" = true ]; then
        ./Engine/Build/BatchFiles/RunUAT.sh $UAT_COMMAND $UAT_ARGS
    else
        ./Engine/Build/BatchFiles/RunUAT.sh $UAT_COMMAND $UAT_ARGS >> "$LOG_FILE" 2>&1
    fi
    
    BUILD_RESULT=$?
    
    # 记录构建结束时间
    echo "构建结束时间: $(date)" >> "$LOG_FILE"
    echo "构建结果: $BUILD_RESULT" >> "$LOG_FILE"
    
    if [ $BUILD_RESULT -ne 0 ]; then
        log_error "构建失败! 错误代码: $BUILD_RESULT"
        log_error "请检查日志文件: $LOG_FILE"
        return $BUILD_RESULT
    fi
    
    log_success "构建完成!"
    return 0
}

# 显示构建结果
show_results() {
    echo ""
    echo "==============================================="
    echo "构建完成!"
    echo "==============================================="
    echo "构建结果摘要:"
    echo "  - 构建目录: $BUILD_DIR"
    echo "  - 构建状态: 成功"
    echo "  - 完成时间: $(date)"
    echo "  - 日志文件: $LOG_FILE"
    echo ""
    
    # 显示构建内容
    if [ -d "$BUILD_DIR" ]; then
        echo "构建内容:"
        find "$BUILD_DIR" -name "*.app" -o -name "UnrealServer" -o -name "UnrealClient" | head -10
        echo ""
    fi
}

# 主函数
main() {
    echo ""
    echo "==============================================="
    echo "UE5.6 Mac 安装构建脚本"
    echo "==============================================="
    echo ""
    
    # 解析参数
    parse_arguments "$@"
    
    # 检查环境
    check_environment
    
    # 检查依赖
    check_dependencies
    
    # 显示配置
    show_config
    
    # 确认构建
    read -p "是否继续构建? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "构建已取消"
        exit 0
    fi
    
    # 设置日志
    setup_logging
    
    # 创建构建目录
    create_build_directory
    
    # 清理构建
    clean_build
    
    # 构建工具
    build_tools
    
    # 准备UAT命令
    prepare_uat_command
    
    # 运行构建
    if run_build; then
        show_results
        log_success "构建脚本执行完成"
    else
        log_error "构建失败，请检查日志文件"
        exit 1
    fi
}

# 运行主函数
main "$@"
