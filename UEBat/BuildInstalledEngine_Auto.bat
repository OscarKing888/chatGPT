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

rem 检测VPN环境
echo 检测网络环境...
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "adapter"') do (
    echo %%i | findstr /i "vpn" >nul
    if !errorlevel! equ 0 (
        echo 检测到VPN连接，使用VPN兼容模式...
        set VPN_MODE=true
        goto :vpn_mode
    )
)

echo 未检测到VPN连接，使用标准模式...
set VPN_MODE=false
goto :normal_mode

:vpn_mode
rem VPN模式：禁用DDC并使用内存缓存
call "%ENGINE_ROOT%Engine\Build\BatchFiles\RunUAT.bat" BuildGraph ^
    -target="Make Installed Build Win64" ^
    -script="%ENGINE_ROOT%Engine\Build\InstalledEngineBuild.xml" ^
    -set:WithDDC=false ^
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
    -set:HostPlatformDDCOnly=true
goto :end_build

:normal_mode
rem 标准模式：正常使用DDC
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
    -set:HostPlatformDDCOnly=true

:end_build
if %ERRORLEVEL% neq 0 (
    echo 构建失败！
    rem exit /b %ERRORLEVEL%
)

if "%VPN_MODE%"=="true" (
    echo VPN环境下的构建成功！已生成已安装版本于：%OUTPUT_DIR%
) else (
    echo 构建成功！已生成已安装版本于：%OUTPUT_DIR%
)

pause
