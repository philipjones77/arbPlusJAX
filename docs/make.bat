@ECHO OFF

set PYTHON=python

if "%1" == "prep" (
  %PYTHON% ..\tools\prepare_sphinx_docs.py
  goto end
)

if "%1" == "html" (
  %PYTHON% ..\tools\build_docs.py --builder html
  goto end
)

if "%1" == "linkcheck" (
  %PYTHON% ..\tools\build_docs.py --builder linkcheck
  goto end
)

if "%1" == "latexpdf" (
  %PYTHON% ..\tools\build_docs.py --builder latexpdf
  goto end
)

if "%1" == "clean" (
  rmdir /S /Q ..\outputs\docs_site 2>NUL
  goto end
)

echo Usage: make.bat ^<prep^|html^|linkcheck^|latexpdf^|clean^>

:end
