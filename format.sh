#!/bin/bash

python -m isort --profile black dotshot dotshotlive.py
python -m black dotshot dotshotlive.py
python -m flake8 dotshot dotshotlive.py
