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

- `<1>` Has shift correction

- `<8-?>` if (Has shift correction) "UTF-8" coded leading channel

- `<n*4>` if (Has shift correction) Shift needed for each channel compared to the leading channel, n = number of
  channels

- `<c*n*b>` if (Has shift correction) Removed samples start + end flattened, c = number of channels, n = number of
  removed samples (max lag), b = bits per sample
  - NOTE: the values are signed two's-complement

- `<1>` Has bias correction

- `<n*8>` if (Has bias correction) DC bias removed from each channel, n = number of channels
  - NOTE: the values are signed two's-complement

- `<1>` Has gain correction

- `<n*12>` if (Has gain correction == 1) Gain correction coefficients (factor) - 1.0, n = number of channels
  - These are unsigned quantized floating point numbers with the range (1 to inf) by for storage purposes 1.0 is
    subtracted since the coefficients are always larger than 1
  - The strongest channel will always have a factor of 1.0 (or 0 quantized)

- `<4>` if (Has gain correction == 1) Gain shift in bits

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

- `<1>` Reserved
    ```
    0 : mandatory value
    1 : reserved
    ```

- `<1>` Block size length:
    ```
    0 : get 8 bit exponent for (2^n) samples
    1 : get 16 bit (blocksize-1)
    ```

- `<0/8>` elif(Block size length bit == 0) blocksize = (2^n) samples

- `<0/16>` if(Block size length bit == 1) 16 bit (blocksize-1)

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

- `<1>` - has coefficients - can mean that this is the main channel, if not the main channel then it is coded
  independently

if (has coefficients):

- `<5>` if(Contains LPC subframes bit == 1) (LPC order) - 1
- `<4>` if(Contains LPC subframes bit == 1) (Quantized linear predictor coefficients' precision in bits)-1.
- `<4>` if(Contains LPC subframes bit == 1) Quantized linear predictor coefficient shift needed in bits
- `<bpc*order>` if(Contains LPC subframes bit == 1) Unencoded predictor coefficients (qlp coeff precision * lpc order) (
  NOTE: the coefficients are signed two's-complement).

else:

- `<0/1>`  Is decorrelated - anly applicable if the frame does not have separate LPC coefficients - whether a localized
  decorrelation was used for this specific subframe

endif

- `<bps*order>` Unencoded warm-up samples (bits-per-sample * lpc order).
- [RESIDUAL](#RESIDUAL) Encoded residual

## RESIDUAL

- `<4>` Starting rice parameter
- `<->` Encoded residual n = frame's blocksize - predictor order 
