@echo off
setlocal

if not "%~1"=="--model" (
    echo Usage: %~nx0 ^<--model model_value^> [--dataset dataset_value] [other_parameters]
    exit /b 1
)

set "model=%~2"

if "%~1" == "--dataset" (
    echo CIAOOO
    set "datasets=%2"
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
        python main.py --dataset="%%d" %*
    )
)

