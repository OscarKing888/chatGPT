@echo off
rem ��������Ŀ¼
set EngineDir=%~dp0..\Engine

rem �����Ƿ�Ϊ UE5��0 ��ʾ UE4��1 ��ʾ UE5��
set IsUE5=1

rem ��Ϸ�ͱ༭����������������
set GameParams=-log -Game -Windowed -ResX=1920 -ResY=1080
set EditorParams=-log

rem ������Ϸ���̲��� -engine������Դ��
set GenGamePrjParams=-game -rocket

rem ���� IsUE5 ��ֵ���������ִ���ļ�����
if "%IsUE5%"=="1" (
    set EngineExe=UnrealEditor.exe
    set EngineExeDebug=UnrealEditor-Win64-Debug.exe
) else (
    set EngineExe=UE4Editor.exe
    set EngineExeDebug=UE4Editor-Win64-Debug.exe
)

rem ��ȡ����Ĳ���
set "prj=%~1"

rem �������Ƿ�Ϊ��
color 0C
if "%prj%"=="" (    
    echo ����Ϊ�գ����ҵ�ǰĿ¼�еĵ�һ�� .uproject �ļ�...
    
    rem ʹ�� for ѭ�����ҵ�һ�� .uproject �ļ������� prj ����
    for %%f in (*.uproject) do (
        set "prj=%%~ff"
        goto :found
    )
    echo ��ǰĿ¼��δ�ҵ� .uproject �ļ���
    exit /b 1
)

:found

rem �������Թ���֤
color 0A
echo ���ò���Ϊ��
echo ====================================================================
echo ����Ŀ¼:        !EngineDir!
echo �����ִ���ļ�:  !EngineExe!
echo ���Կ�ִ���ļ�:  !EngineExeDebug!
echo ���� prj Ϊ:     !prj!
echo ====================================================================
color 0F