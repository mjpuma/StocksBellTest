#!/usr/bin/env python3
"""
Run the full analysis pipeline in order.

  0.py → 1.py → 2.py → 2Bootstrap.py → 2Fig2.py → 3.py → Granger_causality.py

Sets MPLBACKEND=Agg to avoid blocking on interactive plots.
"""
import os
import subprocess
import sys

# Use non-interactive backend for matplotlib
os.environ["MPLBACKEND"] = "Agg"

scripts = [
    "0.py",
    "1.py",
    "2.py",
    "2Bootstrap.py",
    "2Fig2.py",
    "3.py",
    "Granger_causality.py",
]

for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] Running {script}...")
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\n✓ Pipeline completed successfully.")
