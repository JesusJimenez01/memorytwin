@echo off
:: Memory Twin CLI - Acceso rápido desde cualquier lugar
:: Uso: mt search "tu consulta"
::      mt query "tu pregunta"
::      mt stats
::      mt lessons

set MEMORYTWIN_DIR=C:\Users\usuario\Documents\IABigData\MIA\memorytwin
set PYTHON_EXE=C:\Users\usuario\miniforge3\envs\mia\python.exe

cd /d %MEMORYTWIN_DIR%
%PYTHON_EXE% -m memorytwin.escriba.cli %*
