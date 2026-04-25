@echo off
echo =======================================================
echo     Starting S-MAS WebGPU Dashboard 
echo =======================================================
echo.

cd frontend_webgpu

echo Installing dependencies (if needed)...
call npm install

echo.
echo Starting development server...
call npm run dev

pause
