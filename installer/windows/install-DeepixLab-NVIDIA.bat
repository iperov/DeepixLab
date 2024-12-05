@echo off

net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Run as Administrator to continue the installation.
  pause
  exit
)

SET CD=%~dp0
SET TEMP_DIR=%CD%_temp\

SET TEMP_PYTHON_ZIP=%TEMP_DIR%python.zip
SET TEMP_PYTHON_DIR=%TEMP_DIR%python\
SET TEMP_PYTHON_EXE=%TEMP_PYTHON_DIR%python.exe

SET TEMP_REPO_ZIP=%TEMP_DIR%_repo.zip
SET TEMP_REPO=%TEMP_DIR%_repo\
SET TEMP_REPO_INSTALLER_PY=%TEMP_REPO%DeepixLab-master\installer\windows\installer.py

if not exist "%TEMP_DIR%" (
  mkdir "%TEMP_DIR%"
)

powershell -Command Invoke-WebRequest https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip -OutFile "%TEMP_PYTHON_ZIP%"
powershell -Command Expand-Archive '%TEMP_PYTHON_ZIP%' -DestinationPath '%TEMP_PYTHON_DIR%' -Force 

rmdir "%TEMP_REPO%" /s /q 2>nul

powershell -Command Invoke-WebRequest https://github.com/iperov/DeepixLab/archive/refs/heads/master.zip -OutFile "%TEMP_REPO_ZIP%"
powershell -Command Expand-Archive '%TEMP_REPO_ZIP%' -DestinationPath '%TEMP_REPO%' -Force 

"%TEMP_PYTHON_EXE%" "%TEMP_REPO_INSTALLER_PY%" --release-dir "%CD%DeepixLab" --cache-dir "%TEMP_DIR%_cache" --backend cuda

rmdir "%TEMP_DIR%" /s /q 2>nul
