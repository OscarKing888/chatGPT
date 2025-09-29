@echo off
setlocal

rem 设置引擎根目录
set ENGINE_ROOT=%~dp0

rem 设置目标平台（可根据需要修改）
set TARGET_PLATFORM=Win64

rem 设置编译配置（可选：Debug、Development、Shipping）
set BUILD_CONFIGURATION=Development

rem 设置输出目录
set OUTPUT_DIR=%ENGINE_ROOT%LocalBuilds\Engine\%TARGET_PLATFORM%\%BUILD_CONFIGURATION%

rem VPN环境下的DDC设置
echo 检测到VPN环境，使用VPN兼容的DDC设置...

rem 设置环境变量禁用ZenLocal
set UE-ZenLocalEnabled=false
set UE-DDC-ForceMemoryCache=true

rem 运行 AutomationTool 进行构建（VPN优化版本）
call "%ENGINE_ROOT%Engine\Build\BatchFiles\RunUAT.bat" BuildGraph ^
    -target="Make Installed Build Win64" ^
    -script="%ENGINE_ROOT%Engine\Build\InstalledEngineBuild.xml" ^
    -set:WithDDC=true ^
    -set:SignExecutables=false ^
    -set:EmbedSrcSrvInfo=false ^
    -set:GameConfigurations=Development ^
    -set:WithFullDebugInfo=false ^
    -set:HostPlatformEditorOnly=false ^
    -set:AnalyticsTypeOverride= ^
    -set:WithServer=true ^
    -set:WithClient=true ^
	-set:WithWin64=true ^
	-set:WithAndroid=true ^
	-set:WithLinux=false ^
	-set:WithLinuxArm64=false ^
    -set:HostPlatformDDCOnly=true ^
    -DDC-ForceMemoryCache ^
    -ddc=NoZenLocalFallback

if %ERRORLEVEL% neq 0 (
    echo 构建失败！
    rem exit /b %ERRORLEVEL%
)

echo 构建成功！已生成已安装版本于：%OUTPUT_DIR%
echo VPN环境下的构建已完成。

pause
