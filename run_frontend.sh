#!/bin/bash

echo "======================================================="
echo "    Starting S-MAS WebGPU Dashboard (React Frontend)"
echo "======================================================="
echo ""

cd frontend_webgpu || exit 1

echo "Installing dependencies (if needed)..."
npm install

echo ""
echo "Starting development server..."
npm run dev
