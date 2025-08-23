#!/usr/bin/env python3
"""Convenient entry point for DotShot application."""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import and run the main CLI
from main import cli

if __name__ == '__main__':
    cli()
