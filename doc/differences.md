# Differences from FLAC

This file explains the differences from the FLAC format in detail

## STREAM

The "fLaC" marker was replaced by "sTrW"

## METADATA_BLOCK_STREAMINFO

Added:

- `<27>` Total number of frames

Removed:

- `<16>` The minimum block size (in samples) used in the stream.
- `<16>` The maximum block size (in samples) used in the stream. (Minimum blocksize == maximum blocksize) implies a
  fixed-blocksize stream.
  - unnecessary - the blocksize will be stored in the frame header
- `<24>` The minimum frame size (in bytes) used in the stream. May be 0 to imply the value is not known.
- `<24>` The maximum frame size (in bytes) used in the stream. May be 0 to imply the value is not known.
  - completely unnecessary - why was this in FLAC in the first place?

Altered:

- `<20>` Sample rate in Hz. - The maximum sample rate is **NOT** limited by the structure of frame headers.
- `<3>` (number of channels)-1. - This has been changed from a 3 bit value to a UTF-8 coded value ranging from 8 to any
  multiple of 8 bits

## FRAME_HEADER

Added:

- `<1>` Block size length:
    ```
    0 : get 16 bit (blocksize-1)
    1 : get 8 bit exponent for (2^n) samples
    ```

<32>` Size of the frame in bytes (size including the header sync code and the frame footer)

Moved from [SUBFRAME_LPC](#SUBFRAME_LPC):

- `<1>` Contains LPC subframes
    ```
    0 : no
    1 : yes
    ```

- `<0/4>` if(Contains LPC subframes bit == 1) (Quantized linear predictor coefficients' precision in bits)-1.

- `<0/4>` if(Contains LPC subframes bit == 1) Quantized linear predictor coefficient shift needed in bits
  - This has been changed from signed to unsigned

- `<0/bpc*order>` if(Contains LPC subframes bit == 1) Unencoded predictor coefficients (qlp coeff precision * lpc
  order) (
  NOTE: the coefficients are signed two's-complement).

Moved from [SUBFRAME_HEADER](#SUBFRAME_HEADER):

- `<0/5>` if(Contains LPC subframes bit == 1) (LPC order) - 1

Removed:

- `<1>` Reserved:
    ```
    0 : mandatory value
    1 : reserved for future use
    ```
  - unnecessary
- `<1>` Blocking strategy:
    ```
    0 : fixed-blocksize stream; frame header encodes the frame number
    1 : variable-blocksize stream; frame header encodes the sample number
    ```
  - unnecessary - the blocksize is always specified in the frame header
- `<1>` Reserved:
    ```
    0 : mandatory value
    1 : reserved for future use
    ```
- `<4>` Block size in inter-channel samples:
    ```
    0000 : reserved
    0001 : 192 samples
    0010-0101 : 576 * (2^(n-2)) samples, i.e. 576/1152/2304/4608
    0110 : get 8 bit (blocksize-1) from end of header
    0111 : get 16 bit (blocksize-1) from end of header
    1000-1111 : 256 * (2^(n-8)) samples, i.e. 256/512/1024/2048/4096/8192/16384/32768
    ```

- `<4>` Sample rate:
    ```
    0000 : get from STREAMINFO metadata block
    0001 : 88.2kHz
    0010 : 176.4kHz
    0011 : 192kHz
    0100 : 8kHz
    0101 : 16kHz
    0110 : 22.05kHz
    0111 : 24kHz
    1000 : 32kHz
    1001 : 44.1kHz
    1010 : 48kHz
    1011 : 96kHz
    1100 : get 8 bit sample rate (in kHz) from end of header
    1101 : get 16 bit sample rate (in Hz) from end of header
    1110 : get 16 bit sample rate (in tens of Hz) from end of header
    1111 : invalid, to prevent sync-fooling string of 1s
    ```
- `<4>` Channel assignment
    ```
    0000-0111 : (number of independent channels)-1. Where defined, the channel order follows SMPTE/ITU-R recommendations. The assignments are as follows:
        1 channel: mono
        2 channels: left, right
        3 channels: left, right, center
        4 channels: front left, front right, back left, back right
        5 channels: front left, front right, front center, back/surround left, back/surround right
        6 channels: front left, front right, front center, LFE, back/surround left, back/surround right
        7 channels: front left, front right, front center, LFE, back center, side left, side right
        8 channels: front left, front right, front center, LFE, back left, back right, side left, side right
    1000 : left/side stereo: channel 0 is the left channel, channel 1 is the side(difference) channel
    1001 : right/side stereo: channel 0 is the side(difference) channel, channel 1 is the right channel
    1010 : mid/side stereo: channel 0 is the mid(average) channel, channel 1 is the side(difference) channel
    1011-1111 : reserved
    ```

Modified:

- Changed the order of some fields
- Broken byte-alignment in header
- `<0/16>` if(Block size length bit == 0) 16 bit (blocksize-1)
- `<0/8>` elif(Block size length bit == 1) blocksize = (2^n) samples
- `<8-?>` "UTF-8" coded frame number - changed to unspecified length
- `<14>` Sync code '10101010101010'
- changed the sync code

## SUBFRAME_HEADER

Moved to [FRAME](#FRAME_HEADER):

- `<6>` Subframe type:
    ```
    1xxxxx : SUBFRAME_LPC, xxxxx=order-1
    ```

Removed:

- `<1>` Zero bit padding, to prevent sync-fooling string of 1s
-
- `<1+k>` 'Wasted bits-per-sample' flag:
    ```
    0 : no wasted bits-per-sample in source subblock, k=0
    1 : k wasted bits-per-sample in source subblock, k-1 follows, unary coded; e.g. k=3 =>` 001 follows, k=7 =>` 0000001 follows.
    ```

Modified:

- `<6 -> 2>` Subframe type

## SUBFRAME_RAW

- Renamed from **SUBFRAME_VERBATIM**

## SUBFRAME_LPC

Moved to [FRAME](#FRAME_HEADER):

- `<4>` (Quantized linear predictor coefficients' precision in bits)-1 (1111 = invalid).
- `<5>` Quantized linear predictor coefficient shift needed in bits (NOTE: this number is signed two's-complement).
- `<bpc*order>` Unencoded predictor coefficients (qlp coeff precision * lpc order) (NOTE: the coefficients are signed
  two's-complement).
