@echo off
setlocal
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\bin\x64;%PATH%
cd /d "%~dp0"
build\Release\CuRast.exe
