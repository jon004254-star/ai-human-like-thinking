"""
人类思维判定引擎 —— 主入口。

架构: 事件输入 → 语言分析 → 认知分析 → 情境评估 → 年龄调制 → 冲突消解 → 综合判定
安全约束内置于判定流程中，不可绕过。
"""

from typing import Dict, List

from src.core.loader import DataLoader
from src.engines.thinking.datatypes import PersonProfile, EventContext, JudgmentResult
from src.engines.thinking.evaluator import SituationEvaluator
from src.engines.thinking.modulator import AgeModulator
from src.engines.thinking.resolver import ConflictResolver
from src.engines.thinking.safety_guard import SafetyGuard
from src.engines.language.analyzer import LanguageAnalyzer
from src.engines.danger.assessor import DangerAssessor
from src.engines.event_store.store import EventStore
from src.engines.cognitive.engine import CognitiveEngine


class HumanThinkingEngine:
    """
    人类思维判定引擎。

    使用流程:
    1. 构建 PersonProfile（被判定者的年龄/状态等）
    2. 构建 EventContext（事件描述/威胁级别等）
    3. 调用 engine.judge(person, event) 获取判定结果
    4. 检查 result.safety_check_passed
    """

    def __init__(self, data_dir: str = None):
        self.data = DataLoader(data_dir)
        self.situation_evaluator = SituationEvaluator(self.data)
        self.age_modulator = AgeModulator(self.data)
        self.conflict_resolver = ConflictResolver()
        self.safety_guard = SafetyGuard()
        self.language_analyzer = LanguageAnalyzer()
        self.danger_assessor = DangerAssessor()
        self.cognitive_engine = CognitiveEngine()
        self.event_store = EventStore()

    def judge(self, person: PersonProfile, event: EventContext) -> JudgmentResult:
        """执行完整的思维判定流程"""
        notes = []
        stage = self.data.get_stage(person.age)

        # Step 0: 语言分析
        language_result = None
        language_boosts = {}
        if event.user_text:
            language_result = self.language_analyzer.analyze(
                event.user_text, age=person.age
            )
            language_boosts = self.language_analyzer.get_instinct_boosts(language_result)
            notes.append(f"语言分析: 社会过滤分数={language_result.social_filter_score:.2f}")
            notes.append(f"语言分析: 检测到 {len(language_result.detected_signals)} 个本能信号")
            if language_result.danger_flags:
                notes.append(f"语言分析: ⚠ 检测到 {len(language_result.danger_flags)} 个危险信号")
            if language_result.fixation_detected:
                notes.append("语言分析: ⚠ 检测到思维固化/执念")
            if language_result.decoded_deep_intent:
                notes.append(f"深层意图: {language_result.decoded_deep_intent}")

        # Step 0.5: 认知分析
        cognitive_result = None
        if any([
            person.birthplace, person.school_type, person.family_background,
            person.major_life_events, person.social_experience_level,
            person.social_competence, person.language_style,
            person.cognitive_traits, person.value_system,
        ]):
            cognitive_result = self.cognitive_engine.analyze(
                person, user_text=event.user_text
            )
            notes.append(f"认知分析: 世界观置信度={cognitive_result.worldview.confidence if cognitive_result.worldview else 0:.2f}")
            notes.append(f"认知分析: 检测到 {len(cognitive_result.biases)} 个认知偏差, {len(cognitive_result.defense_mechanisms)} 个防御机制")
            notes.append(f"认知分析: 生成 {len(cognitive_result.intent_hypotheses)} 个意图假设")
            notes.append(f"认知分析: 置信度修正={cognitive_result.confidence_modifier:.3f}")
            if cognitive_result.recommendations:
                notes.append(f"认知分析: {len(cognitive_result.recommendations)} 条个性化建议")

        # Step 1: 情境评估 → 本能激活度
        activations = self.situation_evaluator.evaluate(person, event)

        # Step 1.5: 应用语言分析增强值
        if language_boosts:
            for name_en, boost in language_boosts.items():
                for key, act in activations.items():
                    if name_en == key or name_en in key:
                        old = act.current_activation
                        act.current_activation = min(1.0, act.current_activation + boost)
                        if act.current_activation - old > 0.1:
                            if act.state == "normal":
                                act.state = "elevated"
                            elif act.state == "elevated":
                                act.state = "exposed"
                            act.triggering_factors.append(
                                f"语言信号: {name_en}(+{boost:.2f})"
                            )
                        break

        # Step 2: 年龄调制
        modulated_activations, social_coeff = self.age_modulator.modulate(activations, person)
        notes.append(f"社会调制系数: {social_coeff:.2f}")
        notes.append(f"年龄阶段: {stage['name']}")

        # Step 3: 冲突消解 → 主导驱动
        dominant_drivers = self.conflict_resolver.resolve(modulated_activations)

        # Step 4: 预测行为模式和情绪
        behavior = self._predict_behavior(modulated_activations, social_coeff, person)
        emotion = self._predict_emotion(modulated_activations)

        # Step 5: 计算置信度
        base_confidence = self._calculate_confidence(modulated_activations, person, event)
        if language_result and language_result.detected_signals:
            confidence = min(1.0, base_confidence + 0.15)
        else:
            confidence = base_confidence
        cognitive_modifier = cognitive_result.confidence_modifier if cognitive_result else 0.0
        confidence = max(0.05, min(1.0, confidence + cognitive_modifier))

        # Step 6: 危险等级判定
        danger_result = self.danger_assessor.assess(
            instinct_activations=modulated_activations,
            social_modulation_coeff=social_coeff,
            age=person.age,
            stage_name=stage['name'],
            language_danger_flags=language_result.danger_flags if language_result else [],
            fixation_detected=language_result.fixation_detected if language_result else False,
            urgency_from_language=language_result.urgency_level if language_result else 0.0,
        )
        notes.append(f"危险等级: {danger_result.level_name} (分数={danger_result.score:.3f})")
        notes.append(f"首要关切: {danger_result.primary_concern}")
        notes.append(f"建议行动: {danger_result.recommended_action}")

        result = JudgmentResult(
            person_profile=person,
            event=event,
            instinct_activations=modulated_activations,
            social_modulation_coefficient=social_coeff,
            dominant_drivers=dominant_drivers,
            predicted_behavior_pattern=behavior,
            predicted_emotional_response=emotion,
            confidence=confidence,
            safety_check_passed=True,
            language_analysis=language_result,
            danger_assessment=danger_result,
            cognitive_analysis=cognitive_result,
            intent_hypotheses=[{
                "hypothesis": h.hypothesis,
                "confidence": h.confidence,
                "supporting": h.supporting_evidence,
                "opposing": h.opposing_evidence,
                "source": h.source,
            } for h in (cognitive_result.intent_hypotheses if cognitive_result else [])],
            worldview_inference=cognitive_result.worldview if cognitive_result else None,
            cognitive_biases_detected=[b.bias_name for b in (cognitive_result.biases if cognitive_result else [])],
            defense_mechanisms_detected=[d.mechanism_name for d in (cognitive_result.defense_mechanisms if cognitive_result else [])],
            personalized_recommendations=cognitive_result.recommendations if cognitive_result else [],
            cognitive_confidence_modifier=cognitive_modifier,
            notes=notes
        )

        # Step 7: 安全审计（不可绕过）
        result = self.safety_guard.audit(person, event, result)

        # Step 8: 自动存档
        event_id = self.event_store.save(result, person, event)
        notes.append(f"事件已存档: {event_id}")

        return result

    def _predict_behavior(
        self, activations, social_coeff: float, person: PersonProfile
    ) -> str:
        sorted_acts = sorted(
            activations.items(),
            key=lambda x: x[1].current_activation, reverse=True
        )
        top = sorted_acts[:3]
        top_names = [act.instinct_name for _, act in top]

        if social_coeff < 0.3:
            return f"本能主导行为（社会压制弱）——主要驱动: {', '.join(top_names)}"
        elif social_coeff < 0.6:
            return f"本能与社会规则博弈——主要驱动: {', '.join(top_names)}，但受社会规范调制"
        else:
            return f"社会规则主导行为——本能信号微弱: {', '.join(top_names)}仅在背景中"

    def _predict_emotion(self, activations) -> str:
        emotion_map = {
            "fear": "恐惧/焦虑",
            "anger": "愤怒/烦躁",
            "joy": "喜悦/愉快",
            "sadness": "悲伤/低落",
            "disgust": "厌恶/反感",
            "surprise": "惊讶",
            "belongingness": "温暖/孤独",
            "status_seeking": "骄傲/嫉妒",
            "meaning_seeking": "充实/空虚",
        }
        emotions = []
        for name_en, act in activations.items():
            if name_en in emotion_map and act.current_activation > 0.4:
                emotions.append((emotion_map[name_en], act.current_activation))

        emotions.sort(key=lambda x: x[1], reverse=True)
        if not emotions:
            return "中性/平静"
        return " → ".join([f"{e}({w:.2f})" for e, w in emotions[:3]])

    def _calculate_confidence(
        self, activations, person: PersonProfile, event: EventContext
    ) -> float:
        high_activations = sum(
            1 for a in activations.values() if a.current_activation > 0.3
        )
        total = len(activations)
        signal_clarity = min(1.0, high_activations / max(1, total * 0.3))

        info_completeness = 0.5
        if person.health_status != "normal":
            info_completeness += 0.2
        if event.threat_level > 0 or event.social_threat_level > 0:
            info_completeness += 0.15
        if event.event_description:
            info_completeness += 0.15

        confidence = (signal_clarity * 0.6 + info_completeness * 0.4)
        return round(min(1.0, max(0.1, confidence)), 2)


# ===== 便捷函数 =====

def create_person(
    age: float,
    health: str = "normal",
    emotion: str = "neutral",
    social: str = "alone",
    culture: str = "default",
    birthplace: str = "",
    education: str = "",
    school_type: str = "",
    family_background: str = "",
    social_experience: str = "",
    social_competence: str = "",
    major_life_events: List[str] = None,
    language_style: str = "",
    cognitive_traits: List[str] = None,
    value_system: List[str] = None,
    **kwargs
) -> PersonProfile:
    """创建人物画像的便捷函数"""
    return PersonProfile(
        age=age,
        health_status=health,
        emotional_state=emotion,
        social_context=social,
        culture=culture,
        birthplace=birthplace,
        education_level=education,
        school_type=school_type,
        family_background=family_background,
        social_experience_level=social_experience,
        social_competence=social_competence,
        major_life_events=major_life_events or [],
        language_style=language_style,
        cognitive_traits=cognitive_traits or [],
        value_system=value_system or [],
        **kwargs
    )


def create_event(
    event_type: str,
    description: str,
    threat: float = 0.0,
    social_threat: float = 0.0,
    **kwargs
) -> EventContext:
    """创建事件情境的便捷函数"""
    return EventContext(
        event_type=event_type,
        event_description=description,
        threat_level=threat,
        social_threat_level=social_threat,
        **kwargs
    )
