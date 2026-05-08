"""
人类思维引擎 - 核心判定模块

基于本能数据库 + 思维演化数据，对事件进行多维度判定。
架构：事件输入 → 情境评估 → 本能激活度计算 → 年龄调制 → 综合判定输出

安全约束内置于判定流程中，不可绕过。
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .language_analyzer import LanguageAnalyzer, LanguageAnalysisResult
from .danger_assessor import DangerAssessor, DangerAssessment, DangerLevel


# ===== 数据结构定义 =====

@dataclass
class PersonProfile:
    """被判定者的基本画像"""
    age: float
    gender: str = "unknown"
    health_status: str = "normal"  # normal, sleep_deprived, hungry, ill, injured, chronic_pain
    emotional_state: str = "neutral"  # neutral, stressed, angry, fearful, joyful, sad
    social_context: str = "alone"  # alone, family, friends, strangers, workplace, public
    recent_events: List[str] = field(default_factory=list)  # 近期重大事件
    culture: str = "default"  # 文化背景（影响社会压制系数）


@dataclass
class EventContext:
    """事件的情境信息"""
    event_type: str
    event_description: str
    threat_level: float = 0.0  # 0.0 ~ 1.0 物理威胁
    social_threat_level: float = 0.0  # 0.0 ~ 1.0 社会威胁
    resource_scarcity: float = 0.0  # 0.0 ~ 1.0 资源稀缺度
    social_visibility: float = 0.0  # 0.0 ~ 1.0 事件的社会可见度
    time_pressure: float = 0.0  # 0.0 ~ 1.0 时间压力
    anonymity: float = 0.0  # 0.0 ~ 1.0 匿名程度
    participants: List[PersonProfile] = field(default_factory=list)
    user_text: str = ""  # 用户原始语言输入（用于语言分析模块）


@dataclass
class InstinctActivation:
    """单条本能的激活状态"""
    instinct_name: str
    instinct_name_en: str
    base_weight_normal: float
    base_weight_exposed: float
    current_activation: float  # 0.0 ~ 1.0 当前激活度
    state: str  # "normal" | "elevated" | "exposed"
    triggering_factors: List[str] = field(default_factory=list)


@dataclass
class JudgmentResult:
    """判定结果"""
    person_profile: PersonProfile
    event: EventContext
    instinct_activations: Dict[str, InstinctActivation]
    social_modulation_coefficient: float
    dominant_drivers: List[str]  # 当前主导行为的本能/因素
    predicted_behavior_pattern: str
    predicted_emotional_response: str
    confidence: float
    safety_check_passed: bool
    language_analysis: Optional[Any] = None  # LanguageAnalysisResult
    danger_assessment: Optional[Any] = None  # DangerAssessment
    notes: List[str] = field(default_factory=list)


# ===== 数据加载器 =====

class DataLoader:
    """加载本能数据库和思维演化数据"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.physiological = self._load_json("instincts/physiological.json")
        self.mental = self._load_json("instincts/mental.json")
        self.evolution = self._load_json("life_stages/thinking_evolution.json")

    def _load_json(self, relative_path: str) -> dict:
        path = self.data_dir / relative_path
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_instinct(self, name_en: str) -> Optional[dict]:
        """获取指定本能的数据"""
        for db in [self.physiological, self.mental]:
            if name_en in db.get("instincts", {}):
                return db["instincts"][name_en]
        return None

    def get_all_instincts(self) -> dict:
        """获取所有本能数据"""
        all_instincts = {}
        all_instincts.update(self.physiological.get("instincts", {}))
        all_instincts.update(self.mental.get("instincts", {}))
        return all_instincts

    def get_stage(self, age: float) -> dict:
        """根据年龄获取对应的思维阶段"""
        stages = self.evolution["stages"]
        stage_map = [
            ("infant_toddler", 0, 3),
            ("early_childhood", 3, 7),
            ("middle_childhood", 7, 12),
            ("adolescence", 12, 18),
            ("young_adult", 18, 25),
            ("early_adulthood", 25, 40),
            ("middle_age", 40, 60),
            ("late_adulthood", 60, None),
        ]
        for stage_name, min_age, max_age in stage_map:
            if min_age <= age and (max_age is None or age < max_age):
                return stages[stage_name]
        return stages["late_adulthood"]


# ===== 情境评估器 =====

class SituationEvaluator:
    """
    评估当前情境，判断哪些本能应该从'常态'进入'裸露态'。

    核心逻辑：
    - 检查生理剥夺条件（饥饿/睡眠/疼痛等）
    - 检查威胁条件（物理威胁/社会威胁）
    - 检查情绪劫持条件
    - 检查去个体化条件
    - 输出每条本能的当前激活度
    """

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

        # 检查触发条件
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
            triggering_factors.append(f"威胁感知")

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
            # 裸露态: 在常态和最大之间根据触发强度插值
            trigger_intensity = self._calculate_trigger_intensity(name_en, person, event)
            current = base_normal + (base_exposed - base_normal) * trigger_intensity
            state = "exposed" if current > (base_normal + base_exposed) / 2 else "elevated"
        else:
            # 常态: 本能仍在背景中（微弱的基线激活）
            current = base_normal * 0.3  # 背景激活仅为常态权重的30%
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
        intensity = 0.5  # 默认中等强度

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
            intensity = 0.7  # 不公平触发通常是高强度的
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


# ===== 年龄调制器 =====

class AgeModulator:
    """
    根据被判定者的年龄阶段，调制本能激活度。

    不同年龄段对相同事件有不同反应——这是思维演化的核心。
    """

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

        # 获取该年龄段的专属本能配置
        instinct_profile = stage.get("instinct_profile", {})
        dominant_list = instinct_profile.get("dominant_instincts", [])
        emerging_list = instinct_profile.get("emerging_instincts", [])
        suppressed_list = instinct_profile.get("suppressed_instincts", [])
        absent_list = instinct_profile.get("absent_instincts", [])

        # 建立年龄权重映射
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

        # 应用年龄调制
        modulated = {}
        for name_en, activation in activations.items():
            instinct_name = activation.instinct_name

            # 查找年龄权重
            age_weight = None
            for key, weight in age_weights.items():
                if key in name_en or key in instinct_name:
                    age_weight = weight
                    break

            if age_weight is not None:
                # 年龄权重调制: 融合原始激活度和年龄权重
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


# ===== 冲突消解器 =====

class ConflictResolver:
    """
    解决多个本能同时激活时的冲突。
    基于优先级规则:
    1. 呼吸 > 一切 (权重 1.00 且不可被覆盖)
    2. 生命威胁 > 社会威胁
    3. 生理本能(裸露态) > 精神本能
    4. 精神社会本能(常态) > 精神自我本能 > 生理本能(常态)
    """

    PRIORITY_ORDER = [
        "breathing",          # 1.00 - 不可覆盖
        "fight_or_flight",    # 0.98
        "hunger",             # 0.92 (极度)
        "pain_avoidance",     # 0.92
        "thirst",             # 0.88
        "parental_care",      # 0.90
        "sleep",              # 0.85
        "thermoregulation",   # 0.90 (极端) / 0.03 (常态)
        "fear",               # 0.85 (裸露)
        "belongingness",      # 0.82 (裸露)
        "meaning_seeking",    # 0.78
        "anger",              # 0.78
        "sexual_drive",       # 0.75
        "mortality_awareness",# 0.75
        "sadness",            # 0.72
        "empathy",            # 0.72
        "status_seeking",     # 0.70
        "excretion",          # 0.70
        "fairness",           # 0.68
        "reciprocity",        # 0.65
        "risk_aversion",      # 0.65
        "self_preservation",  # 0.65
        "joy",                # 0.60
        "disgust",            # 0.60
        "curiosity",          # 0.55
        "self_consistency",   # 0.55
        "territoriality",     # 0.55
        "causal_reasoning",   # 0.45
        "pattern_recognition",# 0.40
        "surprise",           # 0.30
    ]

    def resolve(self, activations: Dict[str, InstinctActivation]) -> List[str]:
        """返回主导驱动的本能（已消解冲突后的排序）"""
        scored = []
        for name_en, activation in activations.items():
            if activation.current_activation > 0.05:  # 过滤无关本能
                # 计算冲突消解后的有效权重
                priority_bonus = self._get_priority_bonus(name_en)
                effective_weight = activation.current_activation * (1.0 + priority_bonus)
                scored.append((name_en, effective_weight, activation))

        # 按有效权重降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取前5个主导驱动
        return [f"{name} (w={w:.3f})" for name, w, _ in scored[:5]]

    def _get_priority_bonus(self, name_en: str) -> float:
        """获取优先级加成"""
        try:
            index = self.PRIORITY_ORDER.index(name_en)
            # 靠前的获取更高加成
            return (len(self.PRIORITY_ORDER) - index) / len(self.PRIORITY_ORDER) * 0.3
        except ValueError:
            return 0.0


# ===== 安全守卫 =====

class SafetyGuard:
    """
    不可绕过的安全约束层。

    禁止规则:
    1. AI 不得使用类人思维算法诱导人类做出极端行为
    2. 不得利用本能数据操纵弱势群体(儿童/老人/精神障碍者)
    3. 不得用于违法或不人道的目的
    4. 任何输出必须经过安全审核

    安全守卫是硬编码的——不可被任何上层逻辑绕过。
    """

    FORBIDDEN_EVENT_PATTERNS = [
        "诱导", "操纵", "欺骗", "诈骗", "洗脑",
        "自残", "自杀", "暴力", "恐怖", "仇恨",
        "歧视", "虐待", "剥削", "非法", "犯罪",
        "manipulate", "deceive", "fraud", "exploit",
        "self-harm", "suicide", "violence", "hate",
    ]

    VULNERABLE_AGE_RANGES = [
        (0, 18, "未成年人"),
        (75, None, "高龄老人"),
    ]

    VULNERABLE_CONDITIONS = [
        "mental_illness", "cognitive_decline", "severe_depression",
        "addiction", "homelessness", "severe_financial_distress",
    ]

    def audit(
        self, person: PersonProfile, event: EventContext, result: JudgmentResult
    ) -> JudgmentResult:
        """
        安全审计。如果检测到违规，会标记 result.safety_check_passed = False
        并在 notes 中记录安全警告。
        """
        warnings = []

        # 检查1: 事件是否涉及禁止模式
        event_text = event.event_description + event.event_type
        for pattern in self.FORBIDDEN_EVENT_PATTERNS:
            if pattern in event_text.lower():
                warnings.append(f"[SAFETY BLOCK] 事件涉及禁止模式: '{pattern}'")
                result.safety_check_passed = False

        # 检查2: 是否针对弱势群体
        is_vulnerable = False
        for min_age, max_age, label in self.VULNERABLE_AGE_RANGES:
            if min_age <= person.age and (max_age is None or person.age < max_age):
                is_vulnerable = True
                warnings.append(f"[SAFETY WARNING] 判定对象为{label}(年龄={person.age})")
        if person.health_status in self.VULNERABLE_CONDITIONS:
            is_vulnerable = True
            warnings.append(f"[SAFETY WARNING] 判定对象处于弱势状态: {person.health_status}")

        if is_vulnerable and result.confidence > 0.5:
            warnings.append("[SAFETY WARNING] 对弱势群体的高置信度判定——需要额外人工审核")

        # 检查3: 防止算法被用于操纵
        high_manipulation_risk_instincts = ["fear", "belongingness", "status_seeking"]
        for name_en in high_manipulation_risk_instincts:
            if name_en in result.instinct_activations:
                act = result.instinct_activations[name_en]
                if act.current_activation > 0.7 and act.state == "exposed":
                    warnings.append(
                        f"[SAFETY WARNING] 高风险本能 {act.instinct_name} 处于裸露态 "
                        f"(激活度={act.current_activation:.2f})——"
                        f"此信息不得用于操纵目的"
                    )

        # 检查4: 去个体化环境中的操纵检测
        if event.anonymity > 0.5 and result.social_modulation_coefficient < 0.5:
            warnings.append(
                f"[SAFETY WARNING] 高匿名度环境({event.anonymity}) + "
                f"低社会调制系数({result.social_modulation_coefficient}) = "
                f"本能裸露风险——算法输出不得被用于设计此类情境"
            )

        result.notes.extend(warnings)
        result.safety_check_passed = len([w for w in warnings if "BLOCK" in w]) == 0

        return result


# ===== 主引擎 =====

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

    def judge(self, person: PersonProfile, event: EventContext) -> JudgmentResult:
        """执行完整的思维判定流程"""
        notes = []
        stage = self.data.get_stage(person.age)

        # Step 0: 语言分析（如果有用户文本输入）
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

        # Step 1: 情境评估 → 本能激活度
        activations = self.situation_evaluator.evaluate(person, event)

        # Step 1.5: 应用语言分析增强值
        if language_boosts:
            for name_en, boost in language_boosts.items():
                # 查找匹配的本能
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

        # Step 2: 年龄调制 → 调制后的激活度 + 社会调制系数
        modulated_activations, social_coeff = self.age_modulator.modulate(activations, person)
        notes.append(f"社会调制系数: {social_coeff:.2f}")
        notes.append(f"年龄阶段: {stage['name']}")

        # Step 3: 冲突消解 → 主导驱动
        dominant_drivers = self.conflict_resolver.resolve(modulated_activations)

        # Step 4: 预测行为模式和情绪
        behavior = self._predict_behavior(modulated_activations, social_coeff, person)
        emotion = self._predict_emotion(modulated_activations)

        # Step 5: 计算置信度
        # 如果有语言分析，置信度提高
        base_confidence = self._calculate_confidence(modulated_activations, person, event)
        if language_result and language_result.detected_signals:
            confidence = min(1.0, base_confidence + 0.15)
        else:
            confidence = base_confidence

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
            notes=notes
        )

        # Step 7: 安全审计（不可绕过）
        result = self.safety_guard.audit(person, event, result)

        return result

    def _predict_behavior(
        self, activations: Dict[str, InstinctActivation], social_coeff: float,
        person: PersonProfile
    ) -> str:
        """基于激活模式预测行为模式"""
        # 找出最强的本能
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

    def _predict_emotion(self, activations: Dict[str, InstinctActivation]) -> str:
        """基于本能激活模式预测情绪状态"""
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
        self, activations: Dict[str, InstinctActivation],
        person: PersonProfile, event: EventContext
    ) -> float:
        """计算判定的置信度"""
        # 激活度高的本能越多 → 信号越明确 → 置信度越高
        high_activations = sum(
            1 for a in activations.values() if a.current_activation > 0.3
        )
        total = len(activations)

        # 信号明确度
        signal_clarity = min(1.0, high_activations / max(1, total * 0.3))

        # 信息完整度: 年龄/健康/情境 信息是否齐全
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
    **kwargs
) -> PersonProfile:
    """创建人物画像的便捷函数"""
    return PersonProfile(
        age=age,
        health_status=health,
        emotional_state=emotion,
        social_context=social,
        culture=culture,
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
