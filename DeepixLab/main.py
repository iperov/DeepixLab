from __future__ import annotations

import argparse
from pathlib import Path

from core.lib import argparse as lib_argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    run_parser = subparsers.add_parser("run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()
    
    
    faceset_maker_parser = run_subparsers.add_parser("FacesetMaker", help="Run FacesetMaker.")
    faceset_maker_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    def faceset_tools_run(args):
        from FacesetMaker import MxManager, QxManagerApp
        mgr = MxManager(state_path=Path(args.ui_data_dir) / 'FacesetMaker.state')
        app = QxManagerApp(mgr, state_path=Path(args.ui_data_dir) / 'FacesetMaker.ui')
        app.exec()
        app.dispose()
        mgr.dispose()

    faceset_maker_parser.set_defaults(func=faceset_tools_run)
    
    dataset_editor_parser = run_subparsers.add_parser("DatasetEditor", help="Run Dataset Editor.")
    dataset_editor_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    dataset_editor_parser.add_argument('--open-path', required=False, action=lib_argparse.FixPathAction, help="Open dataset path.")
    def faceset_tools_run(args):
        from DatasetEditor import MxManager, QxManagerApp
        mgr = MxManager(open_path_once=Path(args.open_path) if args.open_path is not None else None)
        
        app = QxManagerApp(mgr, state_path=Path(args.ui_data_dir) / 'DatasetEditor.ui')
        app.exec()
        app.dispose()
        mgr.dispose()

    dataset_editor_parser.set_defaults(func=faceset_tools_run)
    
    deepcat_parser = run_subparsers.add_parser("DeepCat", help="Run DeepCat.")
    deepcat_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    deepcat_parser.add_argument('--open-path', required=False, action=lib_argparse.FixPathAction, help="Open .dpc project path.")
    def deepcat_run(args):
        from DeepCat import MxManager, QxManagerApp
        mgr = MxManager(open_path=Path(args.open_path) if args.open_path is not None else None)

        app = QxManagerApp(mgr=mgr, state_path=Path(args.ui_data_dir) / 'DeepCat.ui')
        app.exec()
        app.dispose()

        mgr.dispose()
    deepcat_parser.set_defaults(func=deepcat_run)
    
    
    deepswap_parser = run_subparsers.add_parser("DeepSwap", help="Run DeepSwap.")
    deepswap_parser.add_argument('--ui-data-dir', required=True, action=lib_argparse.FixPathAction, help="UI data directory.")
    deepswap_parser.add_argument('--open-path', required=False, action=lib_argparse.FixPathAction, help="Open .dps project path.")
    def deepswap_run(args):
        from DeepSwap import MxManager, QxManagerApp
        mgr = MxManager(open_path=Path(args.open_path) if args.open_path is not None else None)

        app = QxManagerApp(mgr=mgr, state_path=Path(args.ui_data_dir) / 'DeepSwap.ui')
        app.exec()
        app.dispose()

        mgr.dispose()
    deepswap_parser.set_defaults(func=deepswap_run)
    
    
    def bad_args(args):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

# import code
# code.interact(local=dict(globals(), **locals()))
