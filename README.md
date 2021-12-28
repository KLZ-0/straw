# Straw

[![PyTest Status](https://github.com/KLZ-0/straw/workflows/PyTest/badge.svg)](https://github.com/KLZ-0/straw/actions/)

Straw is a lossless audio codec intended for efficiently storing multichannel audio

# Current state

[![Stability Status](https://img.shields.io/badge/Stability-marginal-red.svg)](https://github.com/KLZ-0/straw/tree/dev)

This project is still a work in progress and as such is not yet ready to be used safely.

The interface and imports are subject to frequent change.

**The use of this library in its current form is strongly discouraged!**

# Design

The base design of Straw is based on [FLAC](https://xiph.org/flac).

Linear prediction (LPC) is used to generate and predict signal which after substraction results in a residual signal
with lower entropy. This residual is then efficiently encoded using rice coding to achieve a similar file size to FLAC
for single channel audio.

In addition to these techniques, Straw also takes advantage of waweform similarities between multiple channels.

In general the differences between channels are a combination of:

- Shift
- Gain
- DC offset

Although other conditions can also affect the signals such as the environment and microphone array material.
