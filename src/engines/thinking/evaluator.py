"""
情境评估器 —— 评估当前情境，判断本能激活度。

核心逻辑：
- 检查生理剥夺条件（饥饿/睡眠/疼痛等）
- 检查威胁条件（物理威胁/社会威胁）
- 检查情绪劫持条件
- 检查去个体化条件
- 输出每条本能的当前激活度
"""

from typing import Dict
from src.core.loader import DataLoader
from src.engines.thinking.datatypes import PersonProfile, EventContext, InstinctActivation


class SituationEvaluator:

    def __init__(self, data_loader: DataLoader):
        self.data = data_loader

    def evaluate(self, person: PersonProfile, event: EventContext) -> Dict[str, InstinctActivation]:
        """评估所有本能在当前情境下的激活度"""
        all_instincts = self.data.get_all_instincts()
        activations = {}

        for name_en, instinct_data in all_instincts.items():
            activation = self._evaluate_single_instinct(
                name_en, instinct_data, person, event
            )
            activations[name_en] = activation

        return activations

    def _evaluate_single_instinct(
        self, name_en: str, data: dict, person: PersonProfile, event: EventContext
    ) -> InstinctActivation:
        """评估单条本能的激活度"""
        base_normal = data.get("weights", {}).get("normal", 0.05)
        base_exposed = data.get("weights", {}).get("exposed", 0.50)

        triggered = False
        triggering_factors = []

        # 生理剥夺检测
        if name_en == "hunger" and person.health_status == "hungry":
            triggered = True
            triggering_factors.append("生理剥夺: 饥饿")
        elif name_en == "sleep" and person.health_status == "sleep_deprived":
            triggered = True
            triggering_factors.append("生理剥夺: 睡眠不足")
        elif name_en == "pain_avoidance" and person.health_status in ("injured", "chronic_pain"):
            triggered = True
            triggering_factors.append("生理剥夺: 疼痛")
        elif name_en == "thirst" and person.health_status == "dehydrated":
            triggered = True
            triggering_factors.append("生理剥夺: 脱水")

        # 威胁检测
        if name_en == "fight_or_flight" and event.threat_level > 0.3:
            triggered = True
            triggering_factors.append(f"物理威胁: level={event.threat_level}")
        if name_en == "fear" and (event.threat_level > 0.2 or event.social_threat_level > 0.5):
            triggered = True
            triggering_factors.append("威胁感知")

        # 社会威胁检测
        if name_en == "status_seeking" and event.social_threat_level > 0.4:
            triggered = True
            triggering_factors.append("社会威胁: 地位受挑战")
        if name_en == "belongingness" and event.social_threat_level > 0.3:
            triggered = True
            triggering_factors.append("社会威胁: 归属受威胁")

        # 公平/互惠检测
        if name_en in ("fairness", "reciprocity") and "不公平" in event.event_description:
            triggered = True
            triggering_factors.append("情境触发: 不公平对待")

        # 情绪状态→本能映射
        emotion_instinct_map = {
            "fearful": ["fear", "fight_or_flight"],
            "angry": ["anger", "fight_or_flight"],
            "sad": ["sadness", "belongingness"],
            "joyful": ["joy"],
            "stressed": ["fear", "anger", "risk_aversion"],
        }
        if person.emotional_state in emotion_instinct_map:
            if name_en in emotion_instinct_map[person.emotional_state]:
                triggered = True
                triggering_factors.append(f"情绪状态: {person.emotional_state}")

        # 近期事件触发
        for event_str in person.recent_events:
            if "丧偶" in event_str or "丧亲" in event_str or "丧失" in event_str:
                if name_en in ("sadness", "belongingness", "mortality_awareness", "meaning_seeking"):
                    triggered = True
                    triggering_factors.append(f"近期事件: {event_str}")
            if "失业" in event_str or "裁员" in event_str or "被解雇" in event_str:
                if name_en in ("status_seeking", "meaning_seeking", "self_consistency"):
                    triggered = True
                    triggering_factors.append(f"近期事件: {event_str}")

        # 去个体化检测
        if event.anonymity > 0.5:
            if name_en in ("sexual_drive", "fight_or_flight", "anger"):
                triggered = True
                triggering_factors.append(f"去个体化: anonymity={event.anonymity}")

        # 计算当前激活度
        if triggered:
            trigger_intensity = self._calculate_trigger_intensity(name_en, person, event)
            current = base_normal + (base_exposed - base_normal) * trigger_intensity
            state = "exposed" if current > (base_normal + base_exposed) / 2 else "elevated"
        else:
            current = base_normal * 0.3
            state = "normal"

        return InstinctActivation(
            instinct_name=data.get("name", name_en),
            instinct_name_en=name_en,
            base_weight_normal=base_normal,
            base_weight_exposed=base_exposed,
            current_activation=round(current, 4),
            state=state,
            triggering_factors=triggering_factors
        )

    def _calculate_trigger_intensity(
        self, name_en: str, person: PersonProfile, event: EventContext
    ) -> float:
        """计算触发强度 (0.0 ~ 1.0)"""
        intensity = 0.5

        if name_en == "hunger":
            intensity = 0.9 if person.health_status == "hungry" else 0.5
        elif name_en == "sleep":
            intensity = 0.85 if person.health_status == "sleep_deprived" else 0.5
        elif name_en == "pain_avoidance":
            if person.health_status == "injured":
                intensity = 0.8
            elif person.health_status == "chronic_pain":
                intensity = 0.6
        elif name_en == "fight_or_flight":
            intensity = max(0.3, event.threat_level)
        elif name_en == "fear":
            intensity = max(event.threat_level, event.social_threat_level * 0.7)
        elif name_en == "status_seeking":
            intensity = max(0.3, event.social_threat_level)
        elif name_en in ("fairness", "reciprocity"):
            intensity = 0.7
        elif name_en == "anger":
            intensity = max(event.social_threat_level, event.threat_level)
        elif name_en == "sadness":
            if person.emotional_state == "sad":
                intensity = 0.7
            elif any("丧" in e for e in person.recent_events):
                intensity = 0.75
        elif name_en == "belongingness":
            if person.emotional_state == "sad":
                intensity = 0.6
            if any("丧" in e for e in person.recent_events):
                intensity = 0.7
        elif name_en in ("mortality_awareness", "meaning_seeking"):
            if any("丧" in e for e in person.recent_events):
                intensity = 0.7

        return min(1.0, max(0.0, intensity))
