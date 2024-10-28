@echo off

set /p dirPath=Enter videos folder: 

if not exist "%dirPath%" (
    echo The directory "%dirPath%" does not exist.
    exit /b 1
)

set "outputDir=%dirPath%\rotated"
if not exist "%outputDir%" (
    mkdir "%outputDir%"
)

for %%f in ("%dirPath%\*.avi") do (    
    ffmpeg -i "%%f" -vf "transpose=1" -c:v h264 -crf 10 -preset fast "%outputDir%\%%~nf.mp4"
)

echo Done !
pause