#!/bin/bash

python -m isort --profile black dotshot
python -m black dotshot
python -m flake8 dotshot
