# Straw

[![PyTest Status](https://github.com/KLZ-0/straw/workflows/PyTest/badge.svg)](https://github.com/KLZ-0/straw/actions/)

Straw is a lossless audio codec intended for efficiently storing multichannel audio

# Current state

[![Stability Status](https://img.shields.io/badge/Stability-mediocre-orange.svg)](https://github.com/KLZ-0/straw/tree/dev)

This project is still a work in progress and as such is not yet ready to be used safely.

The interface and imports are subject to frequent change.

**The use of this library in its current form is strongly discouraged!**

# Installation

Create a Python virtual environment, activate it and install the dependencies

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

For now, straw can be run either using the provided launcher script `main.py` in an activated virtual environment:

```shell
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:src
```

```shell
python3 main.py -i /path/to/source.wav -o /path/to/output.straw
```

Or used as a library, for example:

```python
from straw.encoder import Encoder

e = Encoder()
e.load_files("/path/to/source.wav")
e.create_frames()
e.encode()
e.save_file("/path/to/output.straw")
```
