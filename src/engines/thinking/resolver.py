"""
冲突消解器 —— 解决多个本能同时激活时的优先级冲突。

优先级规则:
1. 呼吸 > 一切
2. 生命威胁 > 社会威胁
3. 生理本能(裸露态) > 精神本能
4. 精神社会本能(常态) > 精神自我本能 > 生理本能(常态)
"""

from typing import Dict, List
from src.engines.thinking.datatypes import InstinctActivation


class ConflictResolver:

    PRIORITY_ORDER = [
        "breathing",
        "fight_or_flight",
        "hunger",
        "pain_avoidance",
        "thirst",
        "parental_care",
        "sleep",
        "thermoregulation",
        "fear",
        "belongingness",
        "meaning_seeking",
        "anger",
        "sexual_drive",
        "mortality_awareness",
        "sadness",
        "empathy",
        "status_seeking",
        "excretion",
        "fairness",
        "reciprocity",
        "risk_aversion",
        "self_preservation",
        "joy",
        "disgust",
        "curiosity",
        "self_consistency",
        "territoriality",
        "causal_reasoning",
        "pattern_recognition",
        "surprise",
    ]

    def resolve(self, activations: Dict[str, InstinctActivation]) -> List[str]:
        """返回主导驱动的本能（已消解冲突后的排序）"""
        scored = []
        for name_en, activation in activations.items():
            if activation.current_activation > 0.05:
                priority_bonus = self._get_priority_bonus(name_en)
                effective_weight = activation.current_activation * (1.0 + priority_bonus)
                scored.append((name_en, effective_weight, activation))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f"{name} (w={w:.3f})" for name, w, _ in scored[:5]]

    def _get_priority_bonus(self, name_en: str) -> float:
        """获取优先级加成"""
        try:
            index = self.PRIORITY_ORDER.index(name_en)
            return (len(self.PRIORITY_ORDER) - index) / len(self.PRIORITY_ORDER) * 0.3
        except ValueError:
            return 0.0
