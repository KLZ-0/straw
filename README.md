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

### Library

The library usage is analogous to soundfile:

```python
import straw

data, sample_rate = straw.read("existing_file.straw")
straw.write("new_file.straw", data, sample_rate)
```

### Standalone encoder/decoder

First activate the virtual environment:

```shell
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:src
```

Encoding:

```shell
python3 main.py -i /path/to/input.wav -o /path/to/output.straw
```

Decoding:

```shell
python3 main.py -d -i /path/to/input.straw -o /path/to/output.wav
```
