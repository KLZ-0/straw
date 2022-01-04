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
