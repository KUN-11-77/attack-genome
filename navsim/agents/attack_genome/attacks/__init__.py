"""连续攻击空间。

包含 10 个攻击模板（设计文档第 5 节）：
    Rain / Snow / Dusk / Dawn / Motion Blur / Digital Noise /
    Light Dust / Dappled Light / Vintage Style / CARLA Style

每个模板都实现统一接口 ``apply(image, strength) -> image``。``strength``
取值范围 ``[0, 1]``，1.0 表示最大干扰。``strength == 0`` 时返回原图。
"""

from navsim.agents.attack_genome.attacks.templates import (
    ATTACK_TEMPLATES,
    AttackTemplate,
    ContinuousAttackSpace,
    list_attack_names,
    get_attack_template,
)
from navsim.agents.attack_genome.attacks.navdream_integration import (
    NavDreamAttackAdapter,
    NavDreamConfig,
)

__all__ = [
    "ATTACK_TEMPLATES",
    "AttackTemplate",
    "ContinuousAttackSpace",
    "list_attack_names",
    "get_attack_template",
    "NavDreamAttackAdapter",
    "NavDreamConfig",
]
