"""nuplan stub — 在 import navsim 之前注入假模块, 绕过 nuplan 不可用问题。

navsim.common.dataclasses 顶层 import 了一堆 nuplan 子模块, 我们没装 nuplan。
这个 stub 注入空但有正确 class 名 的模块, 让 dataclasses 至少能 load。
任何在 stub 里没实现的功能, 调用时会立即报 AttributeError — 适合不需要 nuplan runtime
(只要 load ckpt + forward) 的场景。
"""
from __future__ import annotations
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _mk_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    # 强制 __path__ 让 Python 把它当 package
    mod.__path__ = []  # type: ignore
    sys.modules[name] = mod
    return mod


def install() -> None:
    if "nuplan" in sys.modules:
        return  # already installed

    # ---- nuplan root + 子模块 ----
    _mk_module("nuplan")
    _mk_module("nuplan.common")
    _mk_module("nuplan.common.actor_state")
    _mk_module("nuplan.common.maps")
    _mk_module("nuplan.common.maps.nuplan_map")
    _mk_module("nuplan.database")
    _mk_module("nuplan.database.maps_db")
    _mk_module("nuplan.database.utils")
    _mk_module("nuplan.database.utils.pointclouds")
    _mk_module("nuplan.planning")
    _mk_module("nuplan.planning.simulation")
    _mk_module("nuplan.planning.simulation.observation")
    _mk_module("nuplan.planning.simulation.trajectory")

    # ---- nuplan.common.actor_state.state_representation ----
    sr = _mk_module("nuplan.common.actor_state.state_representation")
    @dataclass
    class StateSE2:
        x: float = 0.0
        y: float = 0.0
        heading: float = 0.0
        def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
            self.x = float(x); self.y = float(y); self.heading = float(heading)
    sr.StateSE2 = StateSE2

    # ---- nuplan.common.maps.abstract_map ----
    am = _mk_module("nuplan.common.maps.abstract_map")
    class AbstractMap:  # type: ignore
        pass
    am.AbstractMap = AbstractMap

    # ---- nuplan.common.maps.maps_datatypes ----
    md = _mk_module("nuplan.common.maps.maps_datatypes")
    class TrafficLightStatuses:  # type: ignore
        UNKNOWN = 0
        GREEN = 1
        YELLOW = 2
        RED = 3
    md.TrafficLightStatuses = TrafficLightStatuses

    # ---- nuplan.common.maps.nuplan_map.map_factory ----
    mf = _mk_module("nuplan.common.maps.nuplan_map.map_factory")
    def get_maps_api(*args, **kwargs):
        raise RuntimeError("nuplan stub: get_maps_api not implemented (we don't need maps for inference)")
    mf.get_maps_api = get_maps_api

    # ---- nuplan.database.maps_db.gpkg_mapsdb ----
    gm = _mk_module("nuplan.database.maps_db.gpkg_mapsdb")
    class MAP_LOCATIONS:  # type: ignore
        BOSTON = "us-ma-boston"
        PITTSBURGH = "us-pa-pittsburgh-hazelwood"
        VEGAS = "us-nv-las-vegas-strip"
        SINGAPORE = "sg-one-north"
    gm.MAP_LOCATIONS = MAP_LOCATIONS

    # ---- nuplan.database.utils.pointclouds.lidar ----
    li = _mk_module("nuplan.database.utils.pointclouds.lidar")
    class LidarPointCloud:  # type: ignore
        @classmethod
        def from_file(cls, *args, **kwargs):
            raise RuntimeError("nuplan stub: from_file not implemented")
    li.LidarPointCloud = LidarPointCloud

    # ---- nuplan.planning.simulation.observation.observation_type ----
    ot = _mk_module("nuplan.planning.simulation.observation.observation_type")
    class DetectionsTracks:  # type: ignore
        pass
    ot.DetectionsTracks = DetectionsTracks

    # ---- nuplan.planning.simulation.trajectory.trajectory_sampling ----
    ts = _mk_module("nuplan.planning.simulation.trajectory.trajectory_sampling")
    @dataclass
    class TrajectorySampling:  # type: ignore
        num_poses: int = 8
        interval_length: float = 0.5
    ts.TrajectorySampling = TrajectorySampling

    # ---- nuplan.common.actor_state.oriented_box (used by transfuser) ----
    ob = _mk_module("nuplan.common.actor_state.oriented_box")
    class OrientedBox:  # type: ignore
        pass
    ob.OrientedBox = OrientedBox

    # ---- nuplan.common.maps.abstract_map: SemanticMapLayer (transfuser) ----
    sml_mod = _mk_module("nuplan.common.maps.abstract_map")
    class SemanticMapLayer:  # type: ignore
        pass
    sml_mod.SemanticMapLayer = SemanticMapLayer

    # ---- nuplan.common.actor_state.tracked_objects_types ----
    tot = _mk_module("nuplan.common.actor_state.tracked_objects_types")
    class TrackedObjectType:  # type: ignore
        VEHICLE = 1
        PEDESTRIAN = 2
        BICYCLE = 3
    tot.TrackedObjectType = TrackedObjectType

    print("  [nuplan stub] installed")


if __name__ == "__main__":
    install()
    # Smoke test: 试着 import
    from nuplan.common.actor_state.state_representation import StateSE2
    s = StateSE2(1.0, 2.0, 0.5)
    print(f"  StateSE2: x={s.x} y={s.y} heading={s.heading}")
