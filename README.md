# Straw

[![PyPI](https://img.shields.io/badge/PyPI-straw--codec-blue.svg)](https://pypi.org/project/straw-codec/)

Straw is a lossless audio codec intended for efficiently storing multichannel audio

# Current state

[![PyTest Status](https://github.com/KLZ-0/straw/workflows/PyTest/badge.svg)](https://github.com/KLZ-0/straw/actions/)
[![Stability Status](https://img.shields.io/badge/Stability-mediocre-yellowgreen.svg)](https://github.com/KLZ-0/straw/tree/dev)

The interface and imports are still subject to change.

# Installation

Create a Python virtual environment, activate it and install the dependencies

#### From PyPI

```shell
# system-wide
pip install straw-codec
# local
pip install --user straw-codec
```

#### From GitHub

```shell
# clone this repo
git clone https://github.com/KLZ-0/straw.git && cd straw
# install
pip install .
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

Encoding:

```shell
straw -i /path/to/input.wav -o /path/to/output.straw
```

Decoding:

```shell
straw -d -i /path/to/input.straw -o /path/to/output.wav
```
