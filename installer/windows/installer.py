import argparse
import json
import os
import re
import shutil
import ssl
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List


class WindowsFolderBuilder:
    """
    Builds stand-alone portable all-in-one python folder for Windows with the project from scratch.
    """

    # Constants
    URL_GET_PIP = r'https://bootstrap.pypa.io/get-pip.py'
    URL_MSVC    = r'https://aka.ms/vs/17/release/vc_redist.x64.exe'
    URL_FFMPEG  = r'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
    URL_ISPC    = r'https://github.com/ispc/ispc/releases/download/v1.24.0/ispc-v1.24.0-windows.zip'
    URL_VSCODE  = r'https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-archive'

    DIRNAME_INTERNAL = '_internal'
    DIRNAME_INTERNAL_CUDA = 'CUDA'
    DIRNAME_INTERNAL_PYTHON = 'python'
    DIRNAME_INTERNAL_FFMPEG = 'ffmpeg'
    DIRNAME_INTERNAL_ISPC = 'ispc'
    DIRNAME_LOCALENV = '_z'
    DIRNAME_TEMP = 't'
    DIRNAME_USERPROFILE = 'u'
    DIRNAME_APPDATA = 'AppData'
    DIRNAME_LOCAL = 'Local'
    DIRNAME_ROAMING = 'Roaming'
    DIRNAME_DESKTOP = 'Desktop'
    DIRNAME_INTERNAL_VSCODE = 'VSCode'

    def __init__(self,  release_path : Path,
                        cache_path : Path,
                        python_ver : str):

        if release_path.exists():
            for _ in range(3):
                print(f'WARNING !!! {release_path} will be removed !')

            input('Press enter to continue.')
            input('Are you sure? Press enter to continue.')

            self.rmdir(release_path)
            
        while release_path.exists():
            time.sleep(0.1)
        release_path.mkdir(parents=True)

        self._release_path = release_path
        self._python_ver = python_ver
        self._cache_path = cache_path
        self._download_cache_path = cache_path / '_dl_cache'
        self._pip_cache_path = cache_path / '_pip_cache'

        self._validate_env()
        self._install_internal()
        self._install_python()
        
    
    @property
    def env(self): return self._env
    @property
    def release_path(self) -> Path: return self._release_path
    @property
    def internal_path(self) -> Path: return self._internal_path
    @property
    def python_path(self) -> Path: return self._python_path
    @property
    def python_site_packages_path(self) -> Path: return self._python_site_packages_path
    @property
    def cuda_bin_path(self) -> Path: return self._cuda_bin_path
    
    def copyfiletree(self, src, dst):
        shutil.copytree(src, dst)

    def copyfile(self, src, dst):
        shutil.copyfile(src, dst)
        
    def install_pip_package(self, pkg_name):
        self.run_python(f'-m pip install {pkg_name}')

    def run(self, exec_name : str, argsline : str|None = None, cwd : Path=None):
        """
        run with current builder environment and wait
        """
        args = exec_name
        if argsline is not None:
            args += f' {argsline}'
        subprocess.Popen(args=f'{exec_name} {argsline}', cwd=str(cwd) if cwd is not None else None, shell=True, env=self._env).wait()

    def run_python(self, argsline, cwd=None):
        """
            cwd(None)   if None , then .python_path
        """
        self.run('python.exe', argsline, cwd if cwd is not None else self._python_path)

    
    def download_cached(self, url : str, progress_bar=True, use_cached=True) -> Path:
        """Download the file to cache or use cached. Returns Path."""
        if progress_bar:
            print(f'Downloading {url}')

        f = None
        try:
            url_request = urllib.request.urlopen(url, context=ssl._create_unverified_context())
            url_size = int( url_request.getheader('content-length') )
            
            cached_filepath = self._download_cache_path / re.sub(r'[<>:"/\\|?*]', '_', url)
            
            if use_cached:
                if cached_filepath.exists():
                    if url_size == cached_filepath.stat().st_size:
                        print(f'Using cached {cached_filepath}')
                    else:
                        use_cached = False
                else:
                    use_cached = False
                    
            if not use_cached:
                cached_filepath.parent.mkdir(parents=True, exist_ok=True)
                
                file_size_dl = 0
                f = open(cached_filepath, 'wb')
                while True:
                    buffer = url_request.read(8192)
                    if not buffer:
                        break

                    f.write(buffer)

                    file_size_dl += len(buffer)

                    if progress_bar:
                        print(f'Downloading {file_size_dl} / {url_size}', end='\r')

        except:
            print(f'Unable to download {url}')
            raise

        if f is not None:
            f.close()
        
        return cached_filepath
            
    def rmdir(self, path):
        os.system('del /F /S /Q "{}" > nul'.format(str(path)))
        os.system('rmdir /S /Q "{}"'.format(str(path)))

    def rmdir_in_all_subdirs(self, path, subdirname):
        for root, dirs, files in os.walk( str(path), topdown=False):
            if subdirname in dirs:
                self.rmdir( Path(root) / subdirname )
    

    def _validate_env(self):
        env = os.environ.copy()

        self._internal_path = self._release_path / self.DIRNAME_INTERNAL
        self._internal_path.mkdir(exist_ok=True, parents=True)

        self._local_env_path = self._internal_path / self.DIRNAME_LOCALENV
        self._local_env_path.mkdir(exist_ok=True, parents=True)

        self._temp_path = self._local_env_path / self.DIRNAME_TEMP
        self._temp_path.mkdir(exist_ok=True, parents=True)

        self._userprofile_path = self._local_env_path / self.DIRNAME_USERPROFILE
        self._userprofile_path.mkdir(exist_ok=True, parents=True)

        self._desktop_path = self._userprofile_path / self.DIRNAME_DESKTOP
        self._desktop_path.mkdir(exist_ok=True, parents=True)

        self._localappdata_path = self._userprofile_path / self.DIRNAME_APPDATA / self.DIRNAME_LOCAL
        self._localappdata_path.mkdir(exist_ok=True, parents=True)

        self._appdata_path = self._userprofile_path / self.DIRNAME_APPDATA / self.DIRNAME_ROAMING
        self._appdata_path.mkdir(exist_ok=True, parents=True)

        self._python_path = self._internal_path / self.DIRNAME_INTERNAL_PYTHON
        self._python_path.mkdir(exist_ok=True, parents=True)

        self._python_site_packages_path = self._python_path / 'Lib' / 'site-packages'
        self._python_site_packages_path.mkdir(exist_ok=True, parents=True)

        self._cuda_path = self._internal_path / self.DIRNAME_INTERNAL_CUDA
        self._cuda_path.mkdir(exist_ok=True, parents=True)

        self._cuda_bin_path = self._cuda_path / 'bin'
        self._cuda_bin_path.mkdir(exist_ok=True, parents=True)

        self._vscode_path = self._internal_path / self.DIRNAME_INTERNAL_VSCODE
        self._ffmpeg_path = self._internal_path / self.DIRNAME_INTERNAL_FFMPEG
        self._ispc_path = self._internal_path / self.DIRNAME_INTERNAL_ISPC

        self._7zip_path = self._temp_path / '7zip'


        env['INTERNAL']     = str(self._internal_path)
        env['LOCALENV']     = str(self._local_env_path)
        env['TMP']          = \
        env['TEMP']         = str(self._temp_path)
        env['HOME']         = \
        env['HOMEPATH']     = \
        env['USERPROFILE']  = str(self._userprofile_path)
        env['DESKTOP']      = str(self._desktop_path)
        env['LOCALAPPDATA'] = str(self._localappdata_path)
        env['APPDATA']      = str(self._appdata_path)
        env['PYTHONHOME']   = ''
        env['PYTHONPATH']   = ''
        env['PYTHON_PATH']  = str(self._python_path)
        env['PYTHONEXECUTABLE']  = \
        env['PYTHON_EXECUTABLE'] = \
        env['PYTHON_BIN_PATH']   = str(self._python_path / 'python.exe')
        env['PYTHONWEXECUTABLE'] = \
        env['PYTHON_WEXECUTABLE'] = str(self._python_path / 'pythonw.exe')
        env['PYTHON_LIB_PATH']    = str(self._python_path / 'Lib' / 'site-packages')
        env['CUDA_PATH']    = str(self._cuda_path)
        env['PATH']   = f"{str(self._cuda_path)};{str(self._python_path)};{str(self._python_path / 'Scripts')};{env['PATH']}"

        if self._pip_cache_path is not None:
            env['PIP_CACHE_DIR'] = str(self._pip_cache_path)

        self._env = env

    def _install_internal(self):

        (self._internal_path / 'setenv.bat').write_text(
fr"""@echo off
SET INTERNAL=%~dp0
SET INTERNAL=%INTERNAL:~0,-1%
SET LOCALENV=%INTERNAL%\{self.DIRNAME_LOCALENV}
SET TMP=%LOCALENV%\{self.DIRNAME_TEMP}
SET TEMP=%TMP%
SET HOME=%LOCALENV%\{self.DIRNAME_USERPROFILE}
SET HOMEPATH=%HOME%
SET USERPROFILE=%HOME%
SET DESKTOP=%HOME%\{self.DIRNAME_DESKTOP}
SET LOCALAPPDATA=%USERPROFILE%\{self.DIRNAME_APPDATA}\{self.DIRNAME_LOCAL}
SET APPDATA=%USERPROFILE%\{self.DIRNAME_APPDATA}\{self.DIRNAME_ROAMING}

SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHON_PATH=%INTERNAL%\python
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHON_EXECUTABLE=%PYTHONEXECUTABLE%
SET PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHONW_EXECUTABLE=%PYTHONWEXECUTABLE%
SET PYTHON_BIN_PATH=%PYTHONEXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET ISPC_PATH=%INTERNAL%\{self.DIRNAME_INTERNAL_ISPC}
SET CUDA_PATH=%INTERNAL%\{self.DIRNAME_INTERNAL_CUDA}
SET CUDA_BIN_PATH=%CUDA_PATH%\bin
SET CUDA_CACHE_PATH=%CUDA_PATH%\Cache
SET QT_QPA_PLATFORM_PLUGIN_PATH=%PYTHON_LIB_PATH%\PyQT6\Qt6\Plugins\platforms

SET PATH=%INTERNAL%\ffmpeg;%PYTHON_PATH%;%CUDA_BIN_PATH%;%PYTHON_PATH%\Scripts;%PATH%
""")
        self.clearenv_bat_path = self._internal_path / 'clearenv.bat'
        self.clearenv_bat_path.write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
rmdir %LOCALENV% /s /q 2>nul
mkdir %LOCALENV%
mkdir %TEMP%
mkdir %USERPROFILE%
mkdir %DESKTOP%
mkdir %LOCALAPPDATA%
mkdir %APPDATA%
""")
        (self._internal_path / 'python_console.bat').write_text(
fr"""
@echo off
cd /D %~dp0
call setenv.bat
cd python
cmd
""")

    def _install_python(self):
        print (f"Installing python {self._python_ver} to {self._python_path}")

        python_url = f'https://www.python.org/ftp/python/{self._python_ver}/python-{self._python_ver}-embed-amd64.zip'
        self.download_and_unzip(python_url, self._python_path)

        # Remove _pth file
        for pth_file in self._python_path.glob("*._pth"):
            pth_file.unlink()

        # pth file content if need some specific pip packages working 
        r"""
        .\
        Lib\site-packages
        python310.zip
        """

        print ("Installing pip.")
        self.run_python( str(self.download_cached(self.URL_GET_PIP)) )

    def _get_7zip_bin_path(self):
        if not self._7zip_path.exists():
            self.download_and_unzip(self.URL_7ZIP, self._7zip_path)
        return self._7zip_path / '7za.exe'

    def cleanup(self):
        print ('Cleanup...')
        subprocess.Popen(args=str(self.clearenv_bat_path), shell=True).wait()
        self.rmdir_in_all_subdirs (self._release_path, '__pycache__')

    def pack_sfx_release(self, archive_name):
        archiver_path = self._get_7zip_bin_path()
        archive_path = self._release_path.parent / (archive_name+'.exe')

        subprocess.Popen(args='"%s" a -t7z -sfx7z.sfx -m0=LZMA2 -mx9 -mtm=off -mmt=8 "%s" "%s"' % ( \
                                str(archiver_path),
                                str(archive_path),
                                str(self._release_path)  ),
                            shell=True).wait()

    def download_and_unzip(self, url, unzip_dirpath, only_files_list : List|None = None):
        """
        Download and unzip entire content to unzip_dirpath

         only_files_list(None)  if specified
                                only first match of these files
                                will be extracted to unzip_dirpath without folder structure
        """
        unzip_dirpath.mkdir(parents=True, exist_ok=True)

        tmp_zippath = self.download_cached(url)

        with zipfile.ZipFile(tmp_zippath, 'r') as zip_ref:
            for entry in zip_ref.filelist:

                if only_files_list is not None:
                    only_files_list = list(only_files_list)
                    
                    if not entry.is_dir():
                        entry_filepath = Path( entry.filename )
                        if entry_filepath.name in only_files_list:
                            only_files_list.remove(entry_filepath.name)
                            (unzip_dirpath / entry_filepath.name).write_bytes ( zip_ref.read(entry) )
                else:
                    entry_outpath = unzip_dirpath / Path(entry.filename)

                    if entry.is_dir():
                        entry_outpath.mkdir(parents=True, exist_ok=True)
                    else:
                        entry_outpath.write_bytes ( zip_ref.read(entry) )

    def install_ispc(self):
        print('Installing ispc.')
        self.download_and_unzip(self.URL_ISPC, self._ispc_path.parent)
        shutil.move(self._ispc_path.parent / Path(self.URL_ISPC).stem, self._ispc_path)

    def install_ffmpeg_binaries(self):
        print('Installing ffmpeg binaries.')
        self._ffmpeg_path.mkdir(exist_ok=True, parents=True)
        self.download_and_unzip(self.URL_FFMPEG, self._ffmpeg_path, only_files_list=['ffmpeg.exe', 'ffprobe.exe'] )
    
    def install_msvc(self, folders : List[str] = None):
        """VC redist"""
        print('Installing MSVC redist.')        
        self.run(str(self.download_cached(self.URL_MSVC)), '/q /norestart')
        
        

    def install_vscode(self, folders : List[str] = None):
        """
        Installs vscode
        """
        print('Installing VSCode.')

        self.download_and_unzip(self.URL_VSCODE, self._vscode_path)
    
        # Create bat
        (self._internal_path  / 'vscode.bat').write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
start "" /D "%~dp0" "%INTERNAL%\{self.DIRNAME_INTERNAL_VSCODE}\Code.exe" --disable-workspace-trust "project.code-workspace"
""")

        # Enable portable mode in VSCode
        (self._vscode_path / 'data').mkdir(exist_ok=True)
        # VSCode config
        (self._vscode_path / 'data' / 'argv.json').write_text('{ "disable-hardware-acceleration": true }')

        # Create vscode project
        if folders is None:
            folders = ['.']

        s_folders = ',\n'.join( f'{{ "path" : "{f}" }}' for f in folders )


        (self._internal_path / 'project.code-workspace').write_text (
fr'''{{
	"folders": [{s_folders}
    ],

	"settings": {{
        "breadcrumbs.enabled": false,
        "debug.showBreakpointsInOverviewRuler": true,
        "diffEditor.ignoreTrimWhitespace": true,
        "extensions.ignoreRecommendations": true,
        "editor.renderWhitespace": "none",
        "editor.fastScrollSensitivity": 10,
		"editor.folding": false,
        "editor.minimap.enabled": false,
		"editor.mouseWheelScrollSensitivity": 3,
		"editor.glyphMargin": false,
        "editor.quickSuggestions": {{"other": false,"comments": false,"strings": false}},
        "editor.trimAutoWhitespace": false,
        "python.linting.pylintArgs": ["--disable=import-error"],
        "editor.lightbulb.enabled": "off",
        "python.languageServer": "Pylance",
        "window.menuBarVisibility": "default",
        "window.zoomLevel": 0,
        "python.analysis.diagnosticSeverityOverrides": {{"reportInvalidTypeForm": "none"}},
        "python.defaultInterpreterPath": "${{env:PYTHON_EXECUTABLE}}",
        "python.linting.enabled": false,
        "python.linting.pylintEnabled": false,
        "python.linting.pylamaEnabled": false,
        "python.linting.pydocstyleEnabled": false,
        "telemetry.enableTelemetry": false,
        "workbench.colorTheme": "Visual Studio Light",
        "workbench.activityBar.visible": true,
		"workbench.editor.tabActionCloseVisibility": false,
		"workbench.editor.tabSizing": "shrink",
		"workbench.editor.highlightModifiedTabs": true,
        "workbench.enableExperiments": false,
        "workbench.sideBar.location": "right",
		"files.exclude": {{
			"**/__pycache__": true,
			"**/.github": true,
			"**/.vscode": true,
			"**/*.dat": true,
			"**/*.h5": true,
            "**/*.npy": true
		}},
	}}
}}
''')
        subprocess.Popen(args='bin\code.cmd --disable-workspace-trust --install-extension ms-python.python', cwd=self._vscode_path, shell=True, env=self._env).wait()
        subprocess.Popen(args='bin\code.cmd --disable-workspace-trust --install-extension ms-python.vscode-pylance', cwd=self._vscode_path, shell=True, env=self._env).wait()
        subprocess.Popen(args='bin\code.cmd --disable-workspace-trust --install-extension ms-python.isort', cwd=self._vscode_path, shell=True, env=self._env).wait()
        subprocess.Popen(args='bin\code.cmd --disable-workspace-trust --install-extension searking.preview-vscode', cwd=self._vscode_path, shell=True, env=self._env).wait()

        # Create VSCode user settings
        vscode_user_settings = self._vscode_path / 'data' / 'user-data' / 'User' / 'settings.json'
        vscode_user_settings.parent.mkdir(parents=True, exist_ok=True)
        vscode_user_settings.write_text( json.dumps( {'update.mode' : 'none'}, indent=4 ) )

    def create_run_python_script(self, script_name : str, internal_relative_path : str, args_str : str):

        (self._release_path / script_name).write_text(
fr"""@echo off
cd /D %~dp0
call {self.DIRNAME_INTERNAL}\setenv.bat
"%PYTHONEXECUTABLE%" {self.DIRNAME_INTERNAL}\{internal_relative_path} {args_str}

if %ERRORLEVEL% NEQ 0 (
    echo Error: %ERRORLEVEL% 
    pause
)
""")

    def create_internal_run_python_script(self, script_name : str, internal_relative_path : str, args_str : str):

        (self._internal_path / script_name).write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
"%PYTHONEXECUTABLE%" {internal_relative_path} {args_str}

if %ERRORLEVEL% NEQ 0 (
    echo Error: %ERRORLEVEL% 
    pause
)
""")

def install_deepixlab(release_dir, cache_dir, python_ver='3.12.8', backend='cuda'):
    builder = WindowsFolderBuilder(release_path=Path(release_dir),
                                   cache_path=Path(cache_dir),
                                   python_ver=python_ver)
                                   
    repo_dir_name = 'github_project'
    repo_path = builder.internal_path / repo_dir_name

    builder.install_msvc()
    
    builder.install_ffmpeg_binaries()
    builder.install_ispc()
    
    # PIP INSTALLATIONS
    builder.install_pip_package('numpy==2.1.3')
    builder.install_pip_package('ziglang')
    builder.install_pip_package('PySide6==6.8.1')
    builder.install_pip_package('opencv-python==4.10.0.84')
    builder.install_pip_package('opencv-contrib-python==4.10.0.84')
    builder.install_pip_package('onnx==1.17.0')

    if backend == 'cuda':
        builder.install_pip_package('onnxruntime-gpu==1.19.2')
        builder.install_pip_package('torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124')#('torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124')
            
        print('Moving CUDA dlls from Torch to shared directory')
        torch_lib_path = builder.python_site_packages_path / 'torch' / 'lib'

        for cu_file in torch_lib_path.glob("**/cu*64*.dll"):
            target = builder.cuda_bin_path / cu_file.name
            print (f'Moving {target}')
            shutil.move (str(cu_file), str(target) )

        for file in torch_lib_path.glob("**/nvrtc*.dll"):
            target = builder.cuda_bin_path / file.name
            print (f'Moving {target}')
            shutil.move (str(file), str(target) )

        for file in torch_lib_path.glob("**/zlibwapi.dll"):
            target = builder.cuda_bin_path / file.name
            print (f'Copying {target}')
            shutil.copy (str(file), str(target) )
    elif backend == 'directml':
        builder.install_pip_package('onnxruntime-directml==1.19.2')
        builder.install_pip_package('torch-directml')
    
    print('Copying repository.')
    builder.copyfiletree(Path(__file__).parent.parent.parent, repo_path)
    builder.rmdir_in_all_subdirs(repo_path, '.git')

    print('Creating files.')
    
    builder.create_internal_run_python_script('setup-compile.bat', f'{repo_dir_name}\\DeepixLab\\setup.py', 'compile"')
    builder.create_internal_run_python_script('setup-split-files.bat', f'{repo_dir_name}\\DeepixLab\\setup.py', 'split_files"')
    builder.create_internal_run_python_script('setup-merge-files.bat', f'{repo_dir_name}\\DeepixLab\\setup.py', 'merge_files"')
    
    builder.run('setup-compile.bat', cwd=builder.internal_path)
    builder.run('setup-merge-files.bat', cwd=builder.internal_path)
    
    builder.create_run_python_script('FacesetMaker.bat',    f'{repo_dir_name}\\DeepixLab\\main.py', 'run FacesetMaker --ui-data-dir "%~dp0_internal"')
    builder.create_run_python_script('DatasetEditor.bat',   f'{repo_dir_name}\\DeepixLab\\main.py', 'run DatasetEditor --ui-data-dir "%~dp0_internal"')
    builder.create_run_python_script('DeepCat.bat',         f'{repo_dir_name}\\DeepixLab\\main.py', 'run DeepCat --ui-data-dir "%~dp0_internal"')
    #builder.create_run_python_script('DeepSwap.bat',        f'{repo_dir_name}\\DeepixLab\\main.py', 'run DeepSwap --ui-data-dir "%~dp0_internal"')

    builder.install_vscode(folders=[f'{repo_dir_name}/DeepixLab',f'{repo_dir_name}'])

    builder.cleanup()


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release-dir', action=fixPathAction, default=None)
    p.add_argument('--cache-dir', action=fixPathAction, default=None)
    p.add_argument('--backend', choices=['cuda', 'directml'], default='cuda')

    args = p.parse_args()

    install_deepixlab(release_dir=args.release_dir,
                      cache_dir=args.cache_dir,
                      backend=args.backend)