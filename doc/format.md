# The .straw file format

Heavily based on the FLAC format

`~` before a value means the feature is not yet implemented in the encoder/decoder

# Detailed format description

## STREAM

- `<32>` "sTrW", the Straw stream marker in ASCII, meaning byte 0 of the stream is 0x73, followed by 0x54 0x72 0x57
- [METADATA_BLOCK](#METADATA_BLOCK) This is the mandatory STREAMINFO metadata block that has the basic properties of the
  stream
- [FRAME+](#FRAME)    One or more audio frames

## METADATA_BLOCK

- [METADATA_BLOCK_HEADER](#METADATA_BLOCK_HEADER)    A block header that specifies the type and size of the metadata
  block data.
- [METADATA_BLOCK_DATA](#METADATA_BLOCK_DATA)

## METADATA_BLOCK_HEADER

- `<1>` Last-metadata-block flag: '1' if this block is the last metadata block before the audio blocks, '0' otherwise.

- `<7>` BLOCK_TYPE
    ```
    0 : STREAMINFO
    ~1 : PADDING
    ~2 : APPLICATION
    ~3 : SEEKTABLE
    ~4 : VORBIS_COMMENT
    ~5 : CUESHEET
    ~6 : PICTURE
    ~7-126 : reserved
    ~127 : invalid, to avoid confusion with a frame sync code
    ```

- `<0/24>` if(BLOCK_TYPE != STREAMINFO) Length (in bytes) of metadata to follow (does not include the size of the
  METADATA_BLOCK_HEADER)

## METADATA_BLOCK_DATA

One of:

- [METADATA_BLOCK_STREAMINFO](#METADATA_BLOCK_STREAMINFO)
- ~METADATA_BLOCK_PADDING
- ~METADATA_BLOCK_APPLICATION
- ~METADATA_BLOCK_SEEKTABLE
- ~METADATA_BLOCK_VORBIS_COMMENT
- ~METADATA_BLOCK_CUESHEET
- ~METADATA_BLOCK_PICTURE

## METADATA_BLOCK_STREAMINFO

- `<20>` Sample rate in Hz. Also, a value of 0 is invalid.

- `<8-?>` "UTF-8" coded (number of channels)-1.

- `<5>` (bits per sample)-1. 4 to 32 bits per sample.

- `<27>` Total number of frames

- `<36>` Total samples in stream. 'Samples' means inter-channel sample, i.e. one second of 44.1Khz audio will have 44100
  samples regardless of the number of channels. A value of zero here means the number of total samples is unknown.

- `<128>` MD5 signature of the unencoded audio data. This allows the decoder to determine if an error exists in the
  audio data even when the error does not result in an invalid bitstream.

- `<8-?>` "UTF-8" coded leading channel

- `<n*4>` Shift needed for each channel compared to the leading channel, n = number of channels

- `<c*n*b>` Removed samples start + end flattened, c = number of channels, n = number of removed samples (max lag), b =
  bits per sample
  - NOTE: the values are signed two's-complement

- `<n*8>` DC bias removed from each channel, n = number of channels
  - NOTE: the values are signed two's-complement

- `<?>` Zero-padding to byte alignment.

### NOTES

The "UTF-8" coding is the same variable length code used to store compressed UCS-2, extended to handle larger input.

## FRAME

- [FRAME_HEADER](#FRAME_HEADER)
- [SUBFRAME+](#SUBFRAME) One SUBFRAME per channel.
- `<?>` Zero-padding to byte alignment.
- [FRAME_FOOTER](#FRAME_FOOTER)

## FRAME_HEADER

- `<14>` Sync code '10101010101010'

- `<1>` Contains LPC subframes
    ```
    0 : no
    1 : yes
    ```

- `<1>` Block size length:
    ```
    0 : get 8 bit exponent for (2^n) samples
    1 : get 16 bit (blocksize-1)
    ```

- `<0/8>` elif(Block size length bit == 0) blocksize = (2^n) samples

- `<0/16>` if(Block size length bit == 1) 16 bit (blocksize-1)

- `<0/5>` if(Contains LPC subframes bit == 1) (LPC order) - 1

- `<0/4>` if(Contains LPC subframes bit == 1) (Quantized linear predictor coefficients' precision in bits)-1.

- `<0/4>` if(Contains LPC subframes bit == 1) Quantized linear predictor coefficient shift needed in bits

- `<0/bpc*order>` if(Contains LPC subframes bit == 1) Unencoded predictor coefficients (qlp coeff precision * lpc
  order) (NOTE: the coefficients are signed two's-complement).

- `<?>` if(Contains LPC subframes bit == 1) Zero-padding to byte alignment.

- `<8-?>`: "UTF-8" coded frame number

- `<32>` Size of the frame in bytes (size including the header sync code and the frame footer)

- `<8>` CRC-8 (polynomial = x^8 + x^2 + x^1 + x^0, initialized with 0) of everything before the crc, including the sync
  code

## FRAME_FOOTER

- `<16>` CRC-16 (polynomial = x^16 + x^15 + x^2 + x^0, initialized with 0) of everything before the crc, back to and
  including the frame header sync code

## SUBFRAME

NOTE: Subframes are not byte-aligned

- [SUBFRAME_HEADER](#SUBFRAME_HEADER)
- [SUBFRAME_DATA](#SUBFRAME_DATA)

## SUBFRAME_HEADER

- `<2>` Subframe type:
    ```
    00 : SUBFRAME_CONSTANT
    01 : SUBFRAME_RAW
    10 : reserved
    11 : SUBFRAME_LPC
    ```

## SUBFRAME_DATA

One of:

- [SUBFRAME_CONSTANT](#SUBFRAME_CONSTANT)
- [SUBFRAME_RAW](#SUBFRAME_RAW)
- [SUBFRAME_LPC](#SUBFRAME_LPC)

The [SUBFRAME_HEADER](#SUBFRAME_HEADER) specifies which one.

## SUBFRAME_CONSTANT

- `<n>` Unencoded constant value of the subframe, n = bits-per-sample.

## SUBFRAME_RAW

- `<n*i>` Unencoded samples of the subframe, n = bits-per-sample, i = frame's blocksize.

## SUBFRAME_LPC

- `<1>`  Is coded - whether a localized deconvolve was used for this specific subframe
- `<bps*order>` Unencoded warm-up samples (bits-per-sample * lpc order).
- [RESIDUAL](#RESIDUAL) Encoded residual

## RESIDUAL

- `<4>` Starting rice parameter
- `<->` Encoded residual n = frame's blocksize - predictor order 
