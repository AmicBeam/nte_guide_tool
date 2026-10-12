#!/usr/bin/env python3
"""Generate static tables without importing the website or initializing its DB."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('overlap_generator', ROOT / 'app/modules/overlap/generator.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules', type=Path, default=generator.MODULE / 'data/rules.json')
    parser.add_argument('--output', type=Path, default=generator.MODULE / 'static/data/tables.json')
    args = parser.parse_args()
    payload = generator.generate_files(rule_path=args.rules, output=args.output)
    print(f"已生成{len(payload['characters'])}名角色、{sum(len(c['rows']) for c in payload['characters'])}条路线：{args.output}")
