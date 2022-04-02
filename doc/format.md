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

- <16> The minimum block size (in samples) used in the stream.
- <16> The maximum block size (in samples) used in the stream. (Minimum blocksize == maximum blocksize) implies a
  fixed-blocksize stream.
- <24> The minimum frame size (in bytes) used in the stream. May be 0 to imply the value is not known.
- <24> The maximum frame size (in bytes) used in the stream. May be 0 to imply the value is not known.
- <20> Sample rate in Hz. Though 20 bits are available, the maximum sample rate is limited by the structure of frame
  headers to 655350Hz. Also, a value of 0 is invalid.
- <3> (number of channels)-1. FLAC supports from 1 to 8 channels
- <5> (bits per sample)-1. FLAC supports from 4 to 32 bits per sample. Currently the reference encoder and decoders only
  support up to 24 bits per sample.
- <36> Total samples in stream. 'Samples' means inter-channel sample, i.e. one second of 44.1Khz audio will have 44100
  samples regardless of the number of channels. A value of zero here means the number of total samples is unknown.
- <128> MD5 signature of the unencoded audio data. This allows the decoder to determine if an error exists in the audio
  data even when the error does not result in an invalid bitstream.

### NOTES

```
FLAC specifies a minimum block size of 16 and a maximum block size of 65535, meaning the bit patterns corresponding to the numbers 0-15 in the minimum blocksize and maximum blocksize fields are invalid.
```

## FRAME

- [FRAME_HEADER](#FRAME_HEADER)
- [SUBFRAME+](#SUBFRAME) One SUBFRAME per channel.
- <?> Zero-padding to byte alignment.
- [FRAME_FOOTER](#FRAME_FOOTER)

## FRAME_HEADER

- <14> Sync code '11111111111110'
- <1> Reserved:
    ```
    0 : mandatory value
    1 : reserved for future use
    ```
- <1> Blocking strategy:
    ```
    0 : fixed-blocksize stream; frame header encodes the frame number
    1 : variable-blocksize stream; frame header encodes the sample number
    ```
- <4> Block size in inter-channel samples:
    ```
    0000 : reserved
    0001 : 192 samples
    0010-0101 : 576 * (2^(n-2)) samples, i.e. 576/1152/2304/4608
    0110 : get 8 bit (blocksize-1) from end of header
    0111 : get 16 bit (blocksize-1) from end of header
    1000-1111 : 256 * (2^(n-8)) samples, i.e. 256/512/1024/2048/4096/8192/16384/32768
    ```
- <4> Sample rate:
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
- <4> Channel assignment
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
- <3> Sample size in bits:
    ```
    000 : get from STREAMINFO metadata block
    001 : 8 bits per sample
    010 : 12 bits per sample
    011 : reserved
    100 : 16 bits per sample
    101 : 20 bits per sample
    110 : 24 bits per sample
    111 : reserved
    ```
- <1> Reserved:
    ```
    0 : mandatory value
    1 : reserved for future use
    ```
- <8-48>: "UTF-8" coded frame number (decoded number is 31 bits)
- <0/8/16> if(blocksize bits == 011x) 8/16 bit (blocksize-1)
- <0/8/16> if(sample rate bits == 11xx) 8/16 bit sample rate
- <8> CRC-8 (polynomial = x^8 + x^2 + x^1 + x^0, initialized with 0) of everything before the crc, including the sync
  code

### NOTES

```
This bit must remain reserved for 0 in order for a FLAC frame's initial 15 bits to be distinguishable from the start of an MPEG audio frame (see also).
The "blocking strategy" bit must be the same throughout the entire stream.
The "blocking strategy" bit determines how to calculate the sample number of the first sample in the frame. If the bit is 0 (fixed-blocksize), the frame header encodes the frame number as above, and the frame's starting sample number will be the frame number times the blocksize. If it is 1 (variable-blocksize), the frame header encodes the frame's starting sample number itself. (In the case of a fixed-blocksize stream, only the last block may be shorter than the stream blocksize; its starting sample number will be calculated as the frame number times the previous frame's blocksize, or zero if it is the first frame).
The "UTF-8" coding used for the sample/frame number is the same variable length code used to store compressed UCS-2, extended to handle larger input.
```

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
