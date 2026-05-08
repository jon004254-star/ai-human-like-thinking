"""
年龄调制器 —— 根据年龄阶段调制本能激活度。

不同年龄段对相同事件有不同反应——这是思维演化的核心。
"""

from typing import Dict, Tuple
from src.core.loader import DataLoader
from src.engines.thinking.datatypes import PersonProfile, InstinctActivation


class AgeModulator:

    def __init__(self, data_loader: DataLoader):
        self.data = data_loader

    def modulate(
        self, activations: Dict[str, InstinctActivation], person: PersonProfile
    ) -> Tuple[Dict[str, InstinctActivation], float]:
        """
        用年龄调制本能激活度。
        返回: (调制后的激活度, 社会调制系数)
        """
        stage = self.data.get_stage(person.age)
        social_coefficient = stage.get("social_modulation_coefficient", 0.5)

        instinct_profile = stage.get("instinct_profile", {})
        dominant_list = instinct_profile.get("dominant_instincts", [])
        emerging_list = instinct_profile.get("emerging_instincts", [])
        suppressed_list = instinct_profile.get("suppressed_instincts", [])
        absent_list = instinct_profile.get("absent_instincts", [])

        age_weights = {}
        for item in dominant_list:
            age_weights[item.get("instinct")] = item.get("weight", 0.7)
        for item in emerging_list:
            age_weights[item.get("instinct")] = item.get("weight", 0.4)
        for item in suppressed_list:
            age_weights[item.get("instinct")] = 0.1
        for item in absent_list:
            if isinstance(item, str):
                age_weights[item] = 0.0
            elif isinstance(item, dict):
                age_weights[item.get("instinct", "")] = 0.0

        modulated = {}
        for name_en, activation in activations.items():
            instinct_name = activation.instinct_name

            age_weight = None
            for key, weight in age_weights.items():
                if key in name_en or key in instinct_name:
                    age_weight = weight
                    break

            if age_weight is not None:
                age_factor = age_weight
                new_activation = activation.current_activation * (0.5 + 0.5 * age_factor)
            else:
                new_activation = activation.current_activation

            modulated[name_en] = InstinctActivation(
                instinct_name=activation.instinct_name,
                instinct_name_en=name_en,
                base_weight_normal=activation.base_weight_normal,
                base_weight_exposed=activation.base_weight_exposed,
                current_activation=round(min(1.0, new_activation), 4),
                state=activation.state,
                triggering_factors=activation.triggering_factors
            )

        return modulated, social_coefficient
