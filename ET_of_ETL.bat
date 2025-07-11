@echo off

cd ScriptPython

python 1_PRE_Analisi_Dei_Dati.py
if errorlevel 1 exit /b

python 2_Pulizia_CSV_GDP.py
if errorlevel 1 exit /b

python 3_Pulizia_CSV_OSD.py
if errorlevel 1 exit /b

python 4_Pulizia_Nazioni_In_Comune.py
if errorlevel 1 exit /b

python 5_POST_Analisi_Dei_Dati.py
if errorlevel 1 exit /b

move "READY_OSD_plus_GDP.csv" ..

cd ..

echo Tutti gli script completati con successo.
pause
