@echo off
setlocal enabledelayedexpansion

:: ����Ƿ��ṩ���㹻�Ĳ���
if "%~1"=="" (
    echo ʹ�÷�����
    echo ���½��з֣�split_pptx.bat �����ļ�·�� ���Ŀ¼·��
    echo ��ҳ�淶Χ�з֣�split_pptx.bat �����ļ�·�� ���Ŀ¼·�� ҳ�淶Χ
    echo.
    echo ʾ����
    echo split_pptx.bat input.pptx output
    echo split_pptx.bat input.pptx output "1-3,5-8,9-10"
    exit /b 1
)

:: ���ò���
set "input_file=%~1"
set "output_dir=%~2"
set "page_ranges=%~3"

:: ��ʾ���յ��Ĳ���
echo �����ļ�: !input_file!
echo ���Ŀ¼: !output_dir!
echo ҳ�淶Χ: !page_ranges!

:: ��������ļ��Ƿ����
if not exist "!input_file!" (
    echo ���������ļ� "!input_file!" ������
    exit /b 1
)

:: ���Python�Ƿ�װ
python --version >nul 2>&1
if errorlevel 1 (
    echo ����δ�ҵ�Python����ȷ���Ѱ�װPython����ӵ�ϵͳ����������
    exit /b 1
)

:: ��������Ƿ�װ
python -c "import pptx" >nul 2>&1
if errorlevel 1 (
    echo ���ڰ�װ��Ҫ������...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ���󣺰�װ����ʧ��
        exit /b 1
    )
)

:: ����Python�ű�
if "!page_ranges!"=="" (
    echo ���ڰ��½��з�PPTX�ļ�...
    python split_pptx.py "!input_file!" "!output_dir!"
) else (
    echo ���ڰ�ָ��ҳ�淶Χ�з�PPTX�ļ�...
    python split_pptx.py "!input_file!" "!output_dir!" --page-ranges="!page_ranges!"
)

if errorlevel 1 (
    echo ���󣺴���PPTX�ļ�ʱ��������
    exit /b 1
)

echo ������ɣ�
pause 