import json
import subprocess
from pathlib import Path


def run(args, pipe_stdin=False, pipe_stdout=False, pipe_stderr=False, quiet_stderr=True) -> subprocess.Popen:
    """
    run ffmpeg process

    returns Popen class if success
    otherwise raise Exception
    """
    args = ['ffmpeg'] + args

    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout else None
    stderr_stream = subprocess.PIPE if pipe_stderr else None

    if quiet_stderr and not pipe_stderr:
        stderr_stream = subprocess.DEVNULL

    return subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream)


def probe(path : Path):
    """Run ffprobe on the specified file and return a JSON representation of the output.

    Raises:
        Exception if ffprobe returns a non-zero exit code,
    """
    args = ['ffprobe', '-show_format', '-show_streams',
            '-of', 'json',
            '-select_streams', 'v',
            str(path)]


    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise Exception(err.decode('utf-8'))

    j = json.loads(out.decode('utf-8'))

    return j


