"""NavDream 攻击适配层。

NavDream 提供了"可控外观操控空间"，设计文档中明确要求把它的输出当
作攻击模板使用。本模块把 NavDream 风格（任意可调外观）的输出接入到
我们自己的连续攻击空间，使我们既可以：
    1. 离线使用内置 10 个攻击模板（``templates.py``）；
    2. 注入任意 NavDream-style 的外部适配器。

本文件本身不直接依赖 NavDream 模型（因为它属于另一个仓库），但提供
了清晰的适配协议，使外部代码可以 plug-in 自己的外观操作函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from navsim.agents.attack_genome.attacks.templates import (
    ATTACK_TEMPLATES,
    AttackTemplate,
    ContinuousAttackSpace,
    DEFAULT_STRENGTHS,
    get_attack_template,
    list_attack_names,
)


ArrayImage = np.ndarray
NavDreamFn = Callable[[ArrayImage, float], ArrayImage]


@dataclass
class NavDreamConfig:
    """NavDream 适配器配置。"""

    # 内部兜底模板（当外部 NavDream 函数未注册时使用）
    fallback_templates: Sequence[str] = field(
        default_factory=lambda: list(ATTACK_TEMPLATES.keys())
    )
    strengths: Sequence[float] = field(default_factory=lambda: DEFAULT_STRENGTHS)
    # 可选 NavDream 模型路径（外部模块按需读取）
    model_path: Optional[str] = None
    # 设备
    device: str = "cpu"


class NavDreamAttackAdapter:
    """NavDream → 我们的 ContinuousAttackSpace 适配器。

    用法 1：完全使用内部兜底模板::

        adapter = NavDreamAttackAdapter()
        attacked = adapter(image, attack_name="Rain", strength=0.6)

    用法 2：替换为外部 NavDream 函数（``register(name, fn)``）::

        def my_night(image, strength):
            ...
        adapter = NavDreamAttackAdapter()
        adapter.register("DuskNightNavDream", my_night)
        attacked = adapter(image, attack_name="DuskNightNavDream", strength=0.6)
    """

    def __init__(self, config: Optional[NavDreamConfig] = None) -> None:
        self.config = config or NavDreamConfig()
        self._registry: Dict[str, AttackTemplate] = {}
        # 把兜底模板注册进来
        for name in self.config.fallback_templates:
            if name in ATTACK_TEMPLATES:
                self._registry[name] = get_attack_template(name)

    # ------------------------------------------------------------------
    # 注册接口
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        fn: NavDreamFn,
        description: str = "",
        replace: bool = False,
    ) -> None:
        if name in self._registry and not replace:
            raise ValueError(
                f"Attack '{name}' already registered. Pass replace=True to override."
            )
        self._registry[name] = AttackTemplate(
            name=name,
            description=description or f"NavDream adapter: {name}",
            apply=fn,
        )

    def available(self) -> List[str]:
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # 调用接口
    # ------------------------------------------------------------------
    def __call__(
        self, image: ArrayImage, attack_name: str, strength: float
    ) -> ArrayImage:
        if attack_name not in self._registry:
            raise KeyError(
                f"Attack '{attack_name}' not registered. "
                f"Available: {self.available()}"
            )
        return self._registry[attack_name](image, strength)

    def build_space(
        self, attack_names: Optional[Sequence[str]] = None
    ) -> ContinuousAttackSpace:
        if attack_names is None:
            attack_names = self.available()
        return ContinuousAttackSpace(
            attack_names=attack_names,
            strengths=list(self.config.strengths),
        )

    @staticmethod
    def default_attack_names() -> List[str]:
        return list_attack_names()
