@echo off
echo =======================================================
echo     Starting S-MAS WebGPU Dashboard (React Frontend)
echo =======================================================
echo.

cd frontend_webgpu

echo Installing dependencies (if needed)...
call npm install

echo.
echo Starting development server...
call npm run dev

pause
