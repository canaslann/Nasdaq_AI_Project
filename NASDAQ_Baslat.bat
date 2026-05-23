@echo off
title NASDAQ Tahmin Uygulamasi
color 0A
echo.
echo  ============================================
echo   NASDAQ Yon Tahmin Sistemi - Can Aslan
echo  ============================================
echo.
echo  [*] Uygulama baslatiliyor...
echo  [*] Tarayici otomatik acilacak.
echo  [*] Kapatmak icin bu pencereyi kapatin.
echo.
cd /d "%~dp0"
streamlit run app.py --server.headless false
pause
