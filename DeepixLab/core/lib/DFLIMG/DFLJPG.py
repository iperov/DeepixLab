from __future__ import annotations

import pickle
import struct
from enum import IntEnum

import cv2
import numpy as np


def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter+fmt_size,) + struct.unpack (fmt, data[counter:counter+fmt_size])


class FaceType(IntEnum):
    #enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100, #no align at all, just embedded faceinfo

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

to_string_dict = { FaceType.HALF : 'half_face',
                   FaceType.MID_FULL : 'midfull_face',
                   FaceType.FULL : 'full_face',
                   FaceType.FULL_NO_ALIGN : 'full_face_no_align',
                   FaceType.WHOLE_FACE : 'whole_face',
                   FaceType.HEAD : 'head',
                   FaceType.HEAD_NO_ALIGN : 'head_no_align',

                   FaceType.MARK_ONLY :'mark_only',
                 }

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() }

class SegIEPolyType(IntEnum):
    EXCLUDE = 0
    INCLUDE = 1


class SegIEPoly():
    def __init__(self, type=None, pts=None, **kwargs):
        self.type = type

        if pts is None:
            pts = np.empty( (0,2), dtype=np.float32 )
        else:
            pts = np.float32(pts)
        self.pts = pts
        self.n_max = self.n = len(pts)

    def dump(self):
        return {'type': int(self.type),
                'pts' : self.get_pts(),
               }

    def identical(self, b):
        if self.n != b.n:
            return False
        return (self.pts[0:self.n] == b.pts[0:b.n]).all()

    def get_type(self):
        return self.type

    def add(self, x, y):
        self.pts = np.append(self.pts[0:self.n], [ ( float(x), float(y) ) ], axis=0).astype(np.float32)
        self.n_max = self.n = self.n + 1

    def undo(self):
        self.n = max(0, self.n-1)
        return self.n

    def redo(self):
        self.n = min(len(self.pts), self.n+1)
        return self.n

    def redo_clip(self):
        self.pts = self.pts[0:self.n]
        self.n_max = self.n

    def insert_pt(self, n, pt):
        if n < 0 or n > self.n:
            raise ValueError("insert_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], pt[None,...].astype(np.float32), self.pts[n:]), axis=0)
        self.n_max = self.n = self.n+1

    def remove_pt(self, n):
        if n < 0 or n >= self.n:
            raise ValueError("remove_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], self.pts[n+1:]), axis=0)
        self.n_max = self.n = self.n-1

    def get_last_point(self):
        return self.pts[self.n-1].copy()

    def get_pts(self):
        return self.pts[0:self.n].copy()

    def get_pts_count(self):
        return self.n

    def set_point(self, id, pt):
        self.pts[id] = pt

    def set_points(self, pts):
        self.pts = np.array(pts)
        self.n_max = self.n = len(pts)

    def mult_points(self, val):
        self.pts *= val



class SegIEPolys():
    def __init__(self):
        self.polys = []

    def identical(self, b):
        polys_len = len(self.polys)
        o_polys_len = len(b.polys)
        if polys_len != o_polys_len:
            return False

        return all ([ a_poly.identical(b_poly) for a_poly, b_poly in zip(self.polys, b.polys) ])

    def add_poly(self, ie_poly_type):
        poly = SegIEPoly(ie_poly_type)
        self.polys.append (poly)
        return poly

    def remove_poly(self, poly):
        if poly in self.polys:
            self.polys.remove(poly)

    def has_polys(self):
        return len(self.polys) != 0

    def get_poly(self, id):
        return self.polys[id]

    def get_polys(self):
        return self.polys

    def get_pts_count(self):
        return sum([poly.get_pts_count() for poly in self.polys])

    def sort(self):
        poly_by_type = { SegIEPolyType.EXCLUDE : [], SegIEPolyType.INCLUDE : [] }

        for poly in self.polys:
            poly_by_type[poly.type].append(poly)

        self.polys = poly_by_type[SegIEPolyType.INCLUDE] + poly_by_type[SegIEPolyType.EXCLUDE]

    def __iter__(self):
        for poly in self.polys:
            yield poly

    def overlay_mask(self, mask):
        h,w,c = mask.shape
        white = (1,)*c
        black = (0,)*c
        for poly in self.polys:
            pts = poly.get_pts().astype(np.int32)
            if len(pts) != 0:
                cv2.fillPoly(mask, [pts], white if poly.type == SegIEPolyType.INCLUDE else black )

    def dump(self):
        return {'polys' : [ poly.dump() for poly in self.polys ] }

    def mult_points(self, val):
        for poly in self.polys:
            poly.mult_points(val)

    @staticmethod
    def load(data=None):
        ie_polys = SegIEPolys()
        if data is not None:
            if isinstance(data, list):
                # Backward comp
                ie_polys.polys = [ SegIEPoly(type=type, pts=pts) for (type, pts) in data ]
            elif isinstance(data, dict):
                ie_polys.polys = [ SegIEPoly(**poly_cfg) for poly_cfg in data['polys'] ]

        ie_polys.sort()

        return ie_polys

class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack ("BB", data[data_counter:data_counter+2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True


                if chunk_size == None: #variable size
                    chunk_size, = struct.unpack (">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c+1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append ({'name' : chunk_name,
                                'm_h' : chunk_m_h,
                                'data' : chunk_data,
                                'ex_data' : chunk_ex_data,
                                })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception (f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename) -> DFLJPG|None:
        try:
            inst = DFLJPG.load_raw (filename)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack (d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack (d, c, "=BBBHHBB")
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id) )
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack (d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            if not inst.has_data():
                return None

            return inst
        except Exception as e:
            #print (f'Exception occured while DFLJPG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):
        data = b""

        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate (self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name' : 'APP15',
                    'm_h' : 0xEF,
                    'data' : pickle.dumps(dict_data),
                    'ex_data' : None,
                    }
        self.chunks.insert (last_app_chunk+1, dflchunk)


        for chunk in self.chunks:
            data += struct.pack ("BB", 0xFF, chunk['m_h'] )
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack (">H", len(chunk_data)+2 )
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    # def get_img(self):
    #     if self.img is None:
    #         self.img = cv2_imread(self.filename)
    #     return self.img

    # def get_shape(self):
    #     if self.shape is None:
    #         img = self.get_img()
    #         if img is not None:
    #             self.shape = img.shape
    #     return self.shape

    # def get_height(self):
    #     for chunk in self.chunks:
    #         if type(chunk) == IHDR:
    #             return chunk.height
    #     return 0

    def get_dict(self):
        return self.dfl_dict

    def set_dict (self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            return self.dfl_dict.get('face_type', FaceType.toString (FaceType.FULL) )
    def set_face_type(self, face_type): self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):            return np.array ( self.dfl_dict['landmarks'] )
    def set_landmarks(self, landmarks): self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      return self.dfl_dict.get ('eyebrows_expand_mod', 1.0)
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  return self.dfl_dict.get ('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get ('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array ( self.dfl_dict.get('source_landmarks', None) )
    def set_source_landmarks(self, source_landmarks):   self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys',None) is not None

    def get_seg_ie_polys(self) -> SegIEPolys|None:
        d = self.dfl_dict.get('seg_ie_polys',None)
        if d is not None:
            d = SegIEPolys.load(d)
        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask',None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        return mask_buf

    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[...,None]

        return img.astype(np.float32) / 255.0


    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip( mask_a*255, 0, 255 ).astype(np.uint8)

        data_max_len = 50000

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100,-1,-1):
                ret, buf = cv2.imencode( '.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] )
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf
