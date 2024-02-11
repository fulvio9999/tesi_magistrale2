@echo off
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<description_value^> ^<--model model_value^> [--dataset dataset_value] [other_parameters]
    exit /b 1
)
set "description=%~1"

if not "%~2"=="--model" (
    echo Usage: %~nx0 ^<description_value^> ^<--model model_value^> [--dataset dataset_value] [other_parameters]
    exit /b 1
)
set "model=%~3"

if "%~4" == "--dataset" (
    echo Datset inserito
    set "datasets=%5"
    set "default=False"
) else (
    set "datasets=elephant tiger fox musk1 musk2 messidor"
    set "default=True"
)
echo Model %model% will run on dataset: %datasets%

if "%default%" == "False" (
    python main.py %*
) else (
    for %%d in (%datasets%) do (
        echo.
        echo Running on dataset %%d --------------------------------------------------------:
        python main.py %* --dataset="%%d"
    )
)
