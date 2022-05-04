#!/bin/bash

rm dist/*.tar.gz
python3 -m pip install --upgrade build
python3 -m build
python3 -m twine upload dist/*.tar.gz
