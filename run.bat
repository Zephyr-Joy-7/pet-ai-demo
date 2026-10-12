
---

### 文件 6：`run.bat`

```bat
@echo off
chcp 65001 >nul
title 宠物 AI Demo
:menu
cls
echo ============================================
echo   宠物 AI Demo 启动器
echo ============================================
echo.
echo   [1] NL2SQL：自然语言查询 pet 表
echo   [2] 构建 RAG 向量库（首次必须先做）
echo   [3] RAG：宠物寄养知识库问答
echo   [4] 退出
echo.
set /p choice=请输入选项（1-4）：

if "%choice%"=="1" goto nl2sql
if "%choice%"=="2" goto build
if "%choice%"=="3" goto rag
if "%choice%"=="4" exit /b

:nl2sql
python nl2sql\main.py
pause
goto menu

:build
python rag\build_index.py
pause
goto menu

:rag
python rag\main.py
pause
goto menu