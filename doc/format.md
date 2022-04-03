# The .straw file format

Heavily based on the FLAC format

`~` before a value means the feature is not yet implemented in the encoder/decoder

# Detailed format description

## STREAM

- <32> "sTrW", the Straw stream marker in ASCII, meaning byte 0 of the stream is 0x73, followed by 0x54 0x72 0x57
- [METADATA_BLOCK](#METADATA_BLOCK) This is the mandatory STREAMINFO metadata block that has the basic properties of the
  stream
- [FRAME+](#FRAME)    One or more audio frames

## METADATA_BLOCK

- [METADATA_BLOCK_HEADER](#METADATA_BLOCK_HEADER)    A block header that specifies the type and size of the metadata
  block data.
- [METADATA_BLOCK_DATA](#METADATA_BLOCK_DATA)

## METADATA_BLOCK_HEADER

- <1> Last-metadata-block flag: '1' if this block is the last metadata block before the audio blocks, '0' otherwise.

- <7> BLOCK_TYPE
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

- <24> Length (in bytes) of metadata to follow (does not include the size of the METADATA_BLOCK_HEADER)

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

- <20> Sample rate in Hz. Also, a value of 0 is invalid.

- <8-?> "UTF-8" coded (number of channels)-1.

- <5> (bits per sample)-1. 4 to 32 bits per sample.

- <3> reserved

- <36> Total samples in stream. 'Samples' means inter-channel sample, i.e. one second of 44.1Khz audio will have 44100
  samples regardless of the number of channels. A value of zero here means the number of total samples is unknown.

- <128> MD5 signature of the unencoded audio data. This allows the decoder to determine if an error exists in the audio
  data even when the error does not result in an invalid bitstream.

### NOTES

The "UTF-8" coding is the same variable length code used to store compressed UCS-2, extended to handle larger input.

## FRAME

- [FRAME_HEADER](#FRAME_HEADER)
- [SUBFRAME+](#SUBFRAME) One SUBFRAME per channel.
- <?> Zero-padding to byte alignment.
- [FRAME_FOOTER](#FRAME_FOOTER)

## FRAME_HEADER

- <15> Sync code '101010101010101'

- <1> Block size length:
    ```
    0 : get 16 bit (blocksize-1)
    1 : get 8 bit exponent for (2^n) samples
    ```

- <0/16> if(Block size length bit == 0) 16 bit (blocksize-1)

- <0/8> elif(Block size length bit == 1) blocksize = (2^n) samples

- <8-?>: "UTF-8" coded frame number

- <32> Size of the frame in bytes (size including the header sync code and the frame footer)

- <8> CRC-8 (polynomial = x^8 + x^2 + x^1 + x^0, initialized with 0) of everything before the crc, including the sync
  code

## FRAME_FOOTER

- <16> CRC-16 (polynomial = x^16 + x^15 + x^2 + x^0, initialized with 0) of everything before the crc, back to and
  including the frame header sync code

## SUBFRAME

- [SUBFRAME_HEADER](#SUBFRAME_HEADER)
- [SUBFRAME_DATA](#SUBFRAME_DATA)

## SUBFRAME_HEADER

- <1> Zero bit padding, to prevent sync-fooling string of 1s
- <6> Subframe type:
    ```
    000000 : SUBFRAME_CONSTANT
    000001 : SUBFRAME_VERBATIM
    00001x : reserved
    0001xx : reserved
    001xxx : if(xxx <= 4) SUBFRAME_FIXED, xxx=order ; else reserved
    01xxxx : reserved
    1xxxxx : SUBFRAME_LPC, xxxxx=order-1
    ```
- <1+k> 'Wasted bits-per-sample' flag:
    ```
    0 : no wasted bits-per-sample in source subblock, k=0
    1 : k wasted bits-per-sample in source subblock, k-1 follows, unary coded; e.g. k=3 => 001 follows, k=7 => 0000001 follows.
    ```

## SUBFRAME_DATA

One of:

- ~SUBFRAME_CONSTANT
- ~SUBFRAME_FIXED
- [SUBFRAME_LPC](#SUBFRAME_LPC)
- ~SUBFRAME_VERBATIM

The SUBFRAME_HEADER specifies which one.

## SUBFRAME_LPC

- <bps*order> Unencoded warm-up samples (frame's bits-per-sample * lpc order).
- <4> (Quantized linear predictor coefficients' precision in bits)-1 (1111 = invalid).
- <5> Quantized linear predictor coefficient shift needed in bits (NOTE: this number is signed two's-complement).
- <bpc*order> Unencoded predictor coefficients (qlp coeff precision * lpc order) (NOTE: the coefficients are signed
  two's-complement).
- [RESIDUAL](#RESIDUAL) Encoded residual

## RESIDUAL

- <4> Starting encoding parameter
    ```
    0000-1110 : Rice parameter.
    1111 : Escape code, meaning the partition is in unencoded binary form using n bits per sample; n follows as a 5-bit number.
    ```
- <-> Encoded residual n = frame's blocksize - predictor order 
