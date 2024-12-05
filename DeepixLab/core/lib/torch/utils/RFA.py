from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Callable, Dict, Generator, List, Sequence


class RFA:
    """Immutable list of layers configuration based on receptive field size"""

    @dataclass(frozen=True)
    class Layer:
        kernel_size : int
        stride : int

    class LayersConfig(List[Layer], list):
        def __init__(self, *args, rfs = ...):
            list.__init__(self, *args)
            self._rfs = rfs

        @cached_property
        def layers_count(self) -> int: return len(self)

        @cached_property
        def rfs(self) -> int:
            rfs = 0
            ts = 1
            for i, layer in enumerate(self):
                if i == 0:
                    rfs = layer.kernel_size
                else:
                    rfs += (layer.kernel_size-1)*ts
                ts *= layer.stride
            return rfs

        @cached_property
        def kernel_size_sum(self) -> int: return sum(l.kernel_size for l in self)
        @cached_property
        def stride_sum(self) -> int: return sum(l.stride for l in self)

        def __repr__(self): return self.__str__()
        def __str__(self):
            return f'LayersConfig, RFS:{self.rfs}, ' + ' '.join(f"({l.kernel_size}, {l.stride})" for l in self)


    @staticmethod
    def create( layers : int|Sequence[int] = 5,
                k_list : Sequence[int] = (3,5,7),
                s_list : Sequence[int] = (1,2),
                prefer_more_layers : bool = True) -> RFA:
        """create new based on desired parameters"""
        return RFA._create(layers, k_list, s_list, prefer_more_layers)

    @staticmethod
    @cache
    def _create(layers : int|Sequence[int],
                k_list : Sequence[int],
                s_list : Sequence[int],
                prefer_more_layers : bool) -> RFA:
        if isinstance(layers, int):
            layers = [*range(1, max(1, layers)+1)]

        d : Dict[int, List[RFA.LayersConfig] ] = {}
        # Iterate over all possible configurations
        # Collect all layers_configs for the same receptive field size
        for layers_count in layers:
            for layers_config in RFA._gen_layers_config(layers_count, k_list, s_list):
                rfs = layers_config.rfs

                if (cfg_ar := d.get(rfs, None)) is None:
                    cfg_ar = d[rfs] = []
                cfg_ar.append(layers_config)

        # Choose best layers_config for every rfs
        new_d = {}
        for rfs, cfg_list in d.items():

            if len(cfg_list) == 1:
                best_cfg = cfg_list[0]
            elif len(cfg_list) > 1:
                best_cfg = cfg_list[0]
                for cfg in cfg_list:
                    if (    prefer_more_layers and cfg.layers_count >  best_cfg.layers_count) or \
                       (not prefer_more_layers and cfg.layers_count <= best_cfg.layers_count) or \
                       (cfg.kernel_size_sum <= best_cfg.kernel_size_sum and cfg.stride_sum >= best_cfg.stride_sum):
                        best_cfg = cfg

            new_d[rfs] = best_cfg

        return RFA(new_d)


    def __init__(self, d : dict = None):
        self._d : Dict[ int, RFA.LayersConfig ] = d if d is not None else {}

    @property
    def min_rfs(self, ) -> int: return sorted(self._d.keys())[0]
    @property
    def max_rfs(self, ) -> int: return sorted(self._d.keys())[-1]

    def nearest(self, rfs : int) -> RFA.LayersConfig:
        return self._d[ sorted(self._d.keys(), key=lambda v: abs(rfs-v) )[0] ]


    def filter(self, func : Callable[ [int, RFA.LayersConfig|None ]]) -> RFA:
        """
            func ( rfs : int, cfg : RFA.LayersConfig)
                return RFA.LayersConfig or None if dismiss this record
        """
        new_d = {}

        for k, layers_cfg in self._d.items():
            if (new_layers_cfg := func(k, layers_cfg)) is not None:
                new_d[k] = new_layers_cfg

        return RFA(new_d)

    def print(self):
        for k in sorted(self._d.keys()):
            cfg = self._d[k]
            print(k,cfg)


    def _gen_layers_config(layer_count : int, k_list, s_list, layer_id : int = 0, ) -> Generator[ RFA.LayersConfig ]:
        if layer_id == layer_count:
            yield RFA.LayersConfig( )
        else:
            for ks in k_list:
                for s in s_list:
                    for sub in RFA._gen_layers_config(layer_count, k_list, s_list, layer_id+1):
                        yield RFA.LayersConfig( (RFA.Layer(kernel_size=ks, stride=s),) + tuple(sub) )

