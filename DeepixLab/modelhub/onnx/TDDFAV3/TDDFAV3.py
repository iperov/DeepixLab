from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from core.lib import facedesc as fd
from core.lib import onnxruntime as lib_ort
from core.lib.image import FImage
from core.lib.math import FAffMat3, FVec2fArray, FVec3Array_like, FVec3fArray


class TDDFAV3:

    @dataclass(frozen=True)
    class ExtractResult:
        anno_lmrks_ysa_range : fd.FAnnoLmrk2DYSARange
        anno_lmrks_2d68 : fd.FAnnoLmrk2D68
        anno_pose : fd.FAnnoPose


    @staticmethod
    def get_available_devices() -> List[lib_ort.DeviceRef]:
        return [lib_ort.get_cpu_device()] + lib_ort.get_avail_gpu_devices()

    def __init__(self, device : lib_ort.DeviceRef ):
        """
        arguments

            device_info    ORTDeviceInfo

                use TDDFAV3.get_available_devices()
                to determine a list of avaliable devices accepted by model

        raises
            Exception
        """
        if device not in TDDFAV3.get_available_devices():
            raise Exception(f'device {device} is not in available devices for TDDFAV3')

        self._sess = sess = lib_ort.InferenceSession_with_device(Path(__file__).parent / 'TDDFAV3.onnx', device)
        self._input_name = sess.get_inputs()[0].name

        self._camera_distance = 10.0
        self._world2view_proj =  np.array([ [1015.0, 0, 0],
                                            [0, 1015.0, 0],
                                            [112.0, 112.0, 1] ], dtype=np.float32)

        self._base_68_pts = np.array([  [-7.31536686e-01,  2.18324587e-01, -5.90363741e-01],
                                        [-7.12017059e-01,  2.05892213e-02, -5.66208839e-01],
                                        [-6.73399091e-01, -1.59576654e-01, -5.48621714e-01],
                                        [-6.34890258e-01, -3.22125107e-01, -5.10903060e-01],
                                        [-5.79798579e-01, -4.96289551e-01, -4.19760287e-01],
                                        [-4.81546730e-01, -6.35660410e-01, -2.58796811e-01],
                                        [-3.65231335e-01, -7.19112933e-01, -6.17607236e-02],
                                        [-2.16505706e-01, -7.86682427e-01,  1.23575926e-01],
                                        [ 2.67814938e-03, -8.24886858e-01,  1.96671784e-01],
                                        [ 2.22356066e-01, -7.84747124e-01,  1.23866975e-01],
                                        [ 3.70815665e-01, -7.19669700e-01, -6.14449978e-02],
                                        [ 4.86512244e-01, -6.38140500e-01, -2.58221686e-01],
                                        [ 5.84084928e-01, -5.00324249e-01, -4.18844312e-01],
                                        [ 6.39234543e-01, -3.27089548e-01, -5.09994268e-01],
                                        [ 6.75870717e-01, -1.63255274e-01, -5.48786998e-01],
                                        [ 7.11227357e-01,  1.79298557e-02, -5.67488015e-01],
                                        [ 7.29938626e-01,  2.17761174e-01, -5.88844717e-01],
                                        [-5.71316302e-01,  4.34932321e-01,  3.62963080e-02],
                                        [-4.88676876e-01,  4.98392701e-01,  1.56470120e-01],
                                        [-3.82955074e-01,  5.19677103e-01,  2.39974141e-01],
                                        [-2.81491995e-01,  5.10921121e-01,  2.91035414e-01],
                                        [-1.89839065e-01,  4.85920191e-01,  3.15223455e-01],
                                        [ 1.87804177e-01,  4.88169491e-01,  3.14644337e-01],
                                        [ 2.79995173e-01,  5.13860941e-01,  2.89511085e-01],
                                        [ 3.81981879e-01,  5.23234010e-01,  2.37666845e-01],
                                        [ 4.89439726e-01,  5.01603365e-01,  1.55025303e-01],
                                        [ 5.72133660e-01,  4.36524928e-01,  3.49660516e-02],
                                        [ 1.76257803e-03,  2.96220154e-01,  3.45770597e-01],
                                        [ 2.21833796e-03,  1.75731063e-01,  4.35879111e-01],
                                        [ 3.29834060e-03,  5.63835837e-02,  5.29183626e-01],
                                        [ 3.25349881e-03, -4.61793244e-02,  5.52442431e-01],
                                        [-1.18324623e-01, -1.37969166e-01,  3.28948617e-01],
                                        [-6.76081181e-02, -1.47517920e-01,  3.74784112e-01],
                                        [ 1.47828390e-03, -1.60410553e-01,  3.95576954e-01],
                                        [ 6.99714422e-02, -1.47369280e-01,  3.74610901e-01],
                                        [ 1.20117076e-01, -1.37369201e-01,  3.28538775e-01],
                                        [-4.30975229e-01,  2.91965753e-01,  1.04033232e-01],
                                        [-3.67296219e-01,  3.31681073e-01,  1.86997056e-01],
                                        [-2.76482165e-01,  3.32692534e-01,  1.89165771e-01],
                                        [-1.91997916e-01,  2.88755804e-01,  1.63525820e-01],
                                        [-2.68532574e-01,  2.67138541e-01,  1.85307682e-01],
                                        [-3.63204300e-01,  2.62323439e-01,  1.62104011e-01],
                                        [ 1.87850699e-01,  2.88736135e-01,  1.60180151e-01],
                                        [ 2.72823513e-01,  3.33718687e-01,  1.85039103e-01],
                                        [ 3.65930617e-01,  3.31698567e-01,  1.84490025e-01],
                                        [ 4.31747049e-01,  2.90704608e-01,  1.03410363e-01],
                                        [ 3.62510055e-01,  2.63633072e-01,  1.61344767e-01],
                                        [ 2.66362846e-01,  2.67252117e-01,  1.83337152e-01],
                                        [-2.52169281e-01, -3.81339163e-01,  2.24057317e-01],
                                        [-1.62668362e-01, -3.21485281e-01,  3.36322904e-01],
                                        [-5.57506531e-02, -2.82682508e-01,  3.96588445e-01],
                                        [ 1.30601926e-03, -2.94665217e-01,  4.02246952e-01],
                                        [ 5.80654554e-02, -2.82752544e-01,  3.96721244e-01],
                                        [ 1.65142193e-01, -3.21170121e-01,  3.35649371e-01],
                                        [ 2.48466194e-01, -3.81282359e-01,  2.22357690e-01],
                                        [ 1.59650698e-01, -4.18565661e-01,  3.16695571e-01],
                                        [ 8.32042322e-02, -4.45940644e-01,  3.59758258e-01],
                                        [ 1.67942536e-03, -4.50652659e-01,  3.69605184e-01],
                                        [-7.98306689e-02, -4.45663661e-01,  3.60788703e-01],
                                        [-1.55356705e-01, -4.18663234e-01,  3.17556500e-01],
                                        [-2.27011800e-01, -3.75459403e-01,  2.31117964e-01],
                                        [-7.30654746e-02, -3.50188583e-01,  3.43461752e-01],
                                        [ 3.90908914e-04, -3.49111587e-01,  3.61925006e-01],
                                        [ 7.40414932e-02, -3.50820452e-01,  3.43598366e-01],
                                        [ 2.32114658e-01, -3.76261741e-01,  2.27939427e-01],
                                        [ 7.26732165e-02, -3.66326064e-01,  3.43837738e-01],
                                        [ 8.06166092e-04, -3.69369358e-01,  3.52605700e-01],
                                        [-7.08143190e-02, -3.65602463e-01,  3.43259454e-01]], dtype=np.float32)

        # eyes and nose indexes
        self._68_nm_idxs =  [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,42,45]

    @property
    def input_size(self): return 224

    def extract(self, img : FImage) -> ExtractResult:
        """
        arguments

            img    FImage

        returns
        """
        img = img.bgr().swap_rb().f32()
        H, W, _ = img.shape

        input_size = self.input_size

        h_scale = H / input_size
        w_scale = W / input_size

        feed_img = img.resize(input_size, input_size).CHW()[None,...]

        w_68_pts, = self._sess.run(None, {self._input_name: feed_img})
        w_68_pts = w_68_pts[0]

        # Project to view pts
        v_68_pts = self._project_w_pts(w_68_pts) * (w_scale, h_scale)

        # Calculate two landmarks so that the alignment is
        # as if the camera is flying on a sphere around the center of the skull.

        # Estimate transform from base face mesh 68 pts to world face
        # Using non moving part of the face
        mat = FAffMat3.estimate(self._base_68_pts[self._68_nm_idxs], w_68_pts[self._68_nm_idxs])

        # Translate center point and up point along Y axis from base face mesh to world face using mat
        # Y values are found manually in order to match lmrks68 align.
        p0c, p0u, p1c, p1u = mat.map([  [0,-0.13, -0.1],
                                        [0, 0.53, -0.1],
                                        [0,-0.13, -0.1+1.0],
                                        [0, 0.53, -0.1+1.0], ])

        # Get dist between center and up 3D points
        p0cp0u_dist = np.linalg.norm(p0u-p0c)
        p1cp1u_dist = np.linalg.norm(p1u-p1c)

        # Add dist to +Y worldcoord
        # and project to view
        vp0c, vp0cu, vp0u, \
        vp1c, vp1cu, vp1u = self._project_w_pts([   p0c, p0c + [0,-p0cp0u_dist,0], p0u,
                                                    p1c, p1c + [0,-p1cp1u_dist,0], p1u,
                                                    ]) * (w_scale, h_scale)

        vp0d = (vp0cu-vp0c).length
        vp1d = (vp1cu-vp1c).length

        vp0n = (vp0u-vp0c).normalize()
        vp1n = (vp1u-vp1c).normalize()

        ysa_range = fd.FAnnoLmrk2DYSARange( FVec2fArray([ vp0c+vp0n*vp0d, vp0c-vp0n*vp0d, vp1c+vp1n*vp1d, vp1c-vp1n*vp1d ]) )

        # Calculate pose
        cp_w, zp_w = mat.map([[0,0,0], [0,0,1]])
        cpzp_w = zp_w-cp_w

        d = cpzp_w.length
        pitch_rad = np.arcsin( cpzp_w[1]/d)
        yaw_rad   = np.arcsin(-cpzp_w[0]/d)
        anno_pose = fd.FAnnoPose(pitch_rad,yaw_rad,0)

        return self.ExtractResult(  anno_lmrks_ysa_range=ysa_range,
                                    anno_lmrks_2d68=fd.FAnnoLmrk2D68(v_68_pts),
                                    anno_pose=anno_pose, )

    def _project_w_pts(self, w_pts : FVec3Array_like) -> FVec2fArray:
        """project world points to view points
        (N, 3) -> (N, 2)
        """
        cam_pts = FVec3fArray(w_pts).as_np() * [1,1,-1]
        cam_pts += [0,0,self._camera_distance]

        view_pts = cam_pts @ self._world2view_proj
        view_pts = view_pts[..., :2] / view_pts[..., 2:]
        view_pts *= [1,-1]
        view_pts += [0,self.input_size]
        return FVec2fArray(view_pts)
