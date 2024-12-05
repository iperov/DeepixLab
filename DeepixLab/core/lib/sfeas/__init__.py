"""
Simple File-Embedded Additive Storage (sfeas).

Simple lib for append and read embedded data by uuid-key at the end of any file.

Typical usage scenarions: embed data to images.

Data blocks at the end of file :
        [Any data ...]
        [BLOCK]
        [BLOCK]
        ...
        [BLOCK]

        BLOCK:
            Data bytes (N)

            UUID (16)

            SIZE of whole block(8)

            MAGIC_NUMBER(8)


Motivation:

Initially, there was a need to add various information to graphic
files such as jpeg or tiff.
Experiments with adding additional meta blocks to these graphic
formats showed that after editing a picture in any graphic
editor and saving it, the added information was completely lost.
In this case, there is no sense to disassemble graphic formats
for this purpose. We may as well just add our own information
to the end of the file.

Designed and developed from scratch by github.com/iperov
"""
from ._sfeas import append, read, remove
