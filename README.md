# Straw

[![PyTest Status](https://github.com/KLZ-0/straw/workflows/PyTest/badge.svg)](https://github.com/KLZ-0/straw/actions/)

Straw is a lossless audio codec intended for efficiently storing multichannel audio

# Current state

[![Stability Status](https://img.shields.io/badge/Stability-marginal-red.svg)](https://github.com/KLZ-0/straw/tree/dev)

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

# Design

The base design of Straw is based on [FLAC](https://xiph.org/flac).

Linear prediction (LPC) is used to generate and predict a signal which substracted from the original signal results in a
residual with lower entropy. This residual is then efficiently encoded using rice coding to achieve a similar file size
to FLAC for single channel audio.

In addition to these techniques, Straw also takes advantage of waweform similarities between multiple channels.

In general the differences between channels are a combination of:

- Shift
- Gain
- DC offset

Although other conditions can also affect the signals such as the environment and microphone array material.
