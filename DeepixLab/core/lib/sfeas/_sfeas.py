import struct
from pathlib import Path
from uuid import UUID


def remove(path : Path|str, uuid : UUID):
    """
    remove all data by UUID from the file

    raise IO errors
    """
    f = open(path, "r+b")

    try:
        c = f.seek(0, 2)
        while c >= 32:
            f.seek(c-32, 0)
            uuid_bytes, block_size, magic_number = struct.unpack("16sQQ", f.read(32))

            if magic_number != 0xFEA5FEA5FEA5FEA5:
                break

            c -= block_size

            if uuid == UUID(bytes=uuid_bytes):
                f.seek(c, 0)
                f.write(bytes(block_size-16))
                break
    except:
        ...

    f.close()

def read(path : Path|str, uuid : UUID) -> bytes|None:
    """
    read last data by UUID from file.

    If no data found or corrupted file -> returns None

    raise IO errors
    """
    f = open(path, "rb")

    data = None
    try:
        c = f.seek(0, 2)
        while c >= 32:
            f.seek(c-32, 0)
            uuid_bytes, block_size, magic_number = struct.unpack("16sQQ", f.read(32))

            if magic_number != 0xFEA5FEA5FEA5FEA5:
                break

            c -= block_size

            if uuid == UUID(bytes=uuid_bytes):
                _, data = f.seek(c, 0), f.read(block_size)
                break
    except:
        ...

    f.close()
    return data

def append(path : Path|str, uuid : UUID, data : bytes):
    """
    raises on error
    """
    with open(path, "a+b") as f:
        f.write(data)
        f.write(struct.pack('16sQQ', uuid.bytes, len(data)+8+16+8, 0xFEA5FEA5FEA5FEA5))


