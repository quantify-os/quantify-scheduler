@ECHO OFF

REM Clean autoapi directory before running sphinx commands
if "%1" == "clean" (
    if exist source\autoapi rmdir /s /q source\autoapi
)

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=source
set BUILDDIR=_build
set SPHINXPROJ=quantify

REM -vv can be appended below to activate sphinx verbose mode
REM For a reference of the different sphinxopts flags,
REM see https://www.sphinx-doc.org/en/master/man/sphinx-build.html
REM
REM We can't supply -W due to https://github.com/jupyter/jupyter-sphinx/issues/182
set SPHINXOPTS=--keep-going -n -w build_errors.log

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "clean" "%2" == "html" goto both
goto help

:clean
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:both
%SPHINXBUILD% -M clean html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end