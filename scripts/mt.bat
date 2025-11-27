@echo off
:: Memory Twin CLI - Acceso rápido desde cualquier lugar
:: Uso: mt search "tu consulta"
::      mt query "tu pregunta"
::      mt stats
::      mt lessons
::      mt setup [ruta]  - Configurar Memory Twin en un proyecto

set MEMORYTWIN_DIR=C:\Users\usuario\Documents\IABigData\MIA\memorytwin
set PYTHON_EXE=C:\Users\usuario\miniforge3\envs\mia\python.exe

%PYTHON_EXE% -m memorytwin.escriba.cli %*
