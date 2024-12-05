import argparse
import importlib
import itertools
import math
import sys
import time
from pathlib import Path

from core import ax

repo_root = Path(__file__).parent
large_files_list = [ (repo_root / 'modelhub' / 'onnx' / 'TDDFAV3' / 'TDDFAV3.onnx', 48*1024*1024),

                    ]

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    compile_parser = subparsers.add_parser("compile", help="Compile all internal libraries.")
    def compile_run(args):
        print('Compiling internal libraries...')

        for i, module_name in enumerate(['common.Graph.FGraph',
                                         'core.lib.hash.hash',
                                         'core.lib.image.FImage',
                                         'core.lib.image.aug.Geo',
                                         'core.lib.image.gen._gen',
                                         'core.lib.image.compute._compute',
                                         ]):
            module = importlib.import_module(module_name)
        
            try:
                module.setup_compile()
                print('Success: ', module_name)
            except Exception as e:
                print('Failure: ', module_name, 'Error: \r\n', e)
                sys.exit(1)

    compile_parser.set_defaults(func=compile_run)

    split_files_parser = subparsers.add_parser("split_files", help="Split large files.")
    def split_files_run(args):
        print('Splitting large files...')

        for filepath, part_size in large_files_list:
            print(f'Splitting {filepath}...')
            if filepath.exists():

                with open(filepath, 'rb') as f:
                    f_size = f.seek(0, 2)
                    n_parts = int( math.ceil(f_size / part_size ) )
                    f.seek(0,0)
                    for n_part in range(n_parts):
                        b = f.read(part_size)
                        part_filepath = filepath.parent / (filepath.name+f'.part{n_part}')
                        part_filepath.write_bytes(b)
            else:
                print(f'{filepath} not found. Skipping.')

    split_files_parser.set_defaults(func=split_files_run)

    merge_files_parser = subparsers.add_parser("merge_files", help="Merge large files.")
    def merge_files_run(args):
        print('Merging large files...')
        for filepath, _ in large_files_list:
            print(f'Merging {filepath}...')

            with open(filepath, 'wb') as f:
                for n_part in itertools.count():
                    part_filepath = filepath.parent / (filepath.name+f'.part{n_part}')
                    if part_filepath.exists():
                        f.write(part_filepath.read_bytes())
                    else:
                        break

    merge_files_parser.set_defaults(func=merge_files_run)


    def bad_args(args):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()