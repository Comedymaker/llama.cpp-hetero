"""Portable entry point; reuse frozen route-specific costs by default."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'workspace/mixed-q2q4-8b-20261021/planner'
sys.path.insert(0, str(SOURCE))
from multibit_planner import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--route', choices=('direct', 'staged'), default='direct')
    parser.add_argument('--cost-table', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    table = args.cost_table or SOURCE / (
        'multibit-final-20261024/direct-n8/cost-table.json' if args.route == 'direct'
        else 'multibit-staged-calibrated-20261024/n8/cost-table.json')
    main(['--route', args.route, '--cost-table', str(table),
          '--output', str(args.output), *remaining])
