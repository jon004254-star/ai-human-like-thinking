"""
危险等级判定模块

综合本能激活模式 + 语言信号 + 年龄阶段 + 社会调制系数，
评估当前情境的危险等级。

危险等级定义:
- NONE:     无明显风险
- LOW:      轻微风险，日常范围
- GUARDED:  需要关注，本能开始裸露
- ELEVATED: 多个本能裸露，需干预
- HIGH:     高风险组合，建议立即关注
- CRITICAL: 紧急——可能发生自伤/伤人等极端行为
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class DangerLevel(Enum):
    NONE = 0
    LOW = 1
    GUARDED = 2
    ELEVATED = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class DangerAssessment:
    """危险判定结果"""
    level: DangerLevel
    level_name: str
    score: float  # 0.0 ~ 1.0 综合风险分数
    risk_categories: Dict[str, float]  # 各风险类别的分数
    primary_concern: str  # 最需要关注的问题
    contributing_factors: List[str]  # 导致当前风险等级的因素
    recommended_action: str  # 建议的响应级别
    urgency: str  # "immediate" | "soon" | "watch" | "none"


class DangerAssessor:
    """
    危险等级判定器

    评估维度:
    1. 本能风险组合——哪些本能同时处于裸露态
    2. 年龄脆弱性——青少年/老人的特殊风险
    3. 社会调制系数——越低风险越高
    4. 语言危险信号——来自 LanguageAnalyzer
    5. 思维固化/执念——单一目标加载过重
    """

    # ===== 高风险本能组合 =====
    DANGEROUS_COMBINATIONS = [
        {
            "name": "性冲动失控风险",
            "instincts": ["sexual_drive", "fear", "sadness"],
            "min_activation": 0.4,
            "min_exposed_count": 2,
            "age_multiplier": {
                "adolescence": 2.0,  # 青少年期加倍
                "young_adult": 1.3,
                "default": 1.0
            },
            "description": "性驱动力+负面情绪→冲动性行为或情感勒索风险",
            "concern": "用户可能在强烈情绪驱动下做出性方面的冲动行为，或使用情感绑架('你不答应我就...')",
        },
        {
            "name": "自我伤害风险",
            "instincts": ["sadness", "meaning_seeking", "fear", "belongingness"],
            "min_activation": 0.5,
            "min_exposed_count": 3,
            "age_multiplier": {
                "adolescence": 1.8,
                "late_adulthood": 1.5,
                "default": 1.0
            },
            "description": "悲伤+意义缺失+恐惧+归属断裂→自我伤害风险",
            "concern": "归属断裂和意义缺失的组合是最危险的心理状态——用户可能认为'没人需要我，活着没意义'",
        },
        {
            "name": "攻击他人风险",
            "instincts": ["anger", "fight_or_flight", "status_seeking"],
            "min_activation": 0.45,
            "min_exposed_count": 2,
            "age_multiplier": {
                "adolescence": 1.6,
                "default": 1.0
            },
            "description": "愤怒+战斗反应+地位威胁→攻击行为风险",
            "concern": "地位受到威胁且愤怒处于裸露态——可能通过攻击行为'夺回'地位",
        },
        {
            "name": "被操纵/剥削风险",
            "instincts": ["belongingness", "status_seeking", "fear"],
            "min_activation": 0.5,
            "min_exposed_count": 2,
            "age_multiplier": {
                "adolescence": 2.0,
                "late_adulthood": 1.8,
                "default": 1.0
            },
            "description": "归属需求+地位焦虑+恐惧→极易被利用",
            "concern": "用户当前极度渴望被接受和证明价值——这使ta对'给予认可的人'几乎没有抵抗力",
        },
        {
            "name": "执念/跟踪风险",
            "instincts": ["sexual_drive", "belongingness", "self_consistency"],
            "min_activation": 0.45,
            "min_exposed_count": 2,
            "age_multiplier": {
                "adolescence": 2.0,
                "default": 1.0
            },
            "description": "性驱动+归属需求+自我叙事绑定→可能发展成执念/跟踪行为",
            "concern": "对方被加载了过重的人生意义——被拒后可能无法接受并持续纠缠",
        },
        {
            "name": "极端行为风险（毁灭模式）",
            "instincts": ["meaning_seeking", "mortality_awareness", "fight_or_flight"],
            "min_activation": 0.5,
            "min_exposed_count": 2,
            "age_multiplier": {
                "default": 1.0
            },
            "description": "意义丧失+死亡意识+战斗反应→'豁出去了'的极端行为",
            "concern": "'反正无所谓了'+'不如拼了'的心理状态——可能导致同归于尽或毁灭式行为",
        },
    ]

    # 年龄风险系数
    AGE_RISK_FACTORS = {
        "infant_toddler": 0.8,   # 婴幼儿需要成人监护
        "early_childhood": 0.9,  # 儿童需要成人监护
        "middle_childhood": 0.8, # 儿童需要成人监护
        "adolescence": 1.8,      # 青少年——前额叶未成熟+性驱动力+同伴压力
        "young_adult": 1.2,      # 新兴成年期——仍有乐观偏差
        "early_adulthood": 1.0,  # 成人——基准风险
        "middle_age": 1.1,       # 中年——中年危机可能增加风险
        "late_adulthood": 1.4,   # 老年——社交孤立+认知下降
    }

    def assess(
        self,
        instinct_activations: Dict,
        social_modulation_coeff: float,
        age: float,
        stage_name: str,
        language_danger_flags: List[str],
        fixation_detected: bool,
        urgency_from_language: float,
    ) -> DangerAssessment:
        """
        综合评估危险等级
        """
        risk_categories = {}
        contributing_factors = []

        # 1. 检查高风险本能组合
        for combo in self.DANGEROUS_COMBINATIONS:
            combo_score = self._evaluate_combo(
                combo, instinct_activations, stage_name, age
            )
            if combo_score > 0:
                risk_categories[combo["name"]] = combo_score
                if combo_score > 0.3:
                    contributing_factors.append(
                        f"高风险本能组合 '{combo['name']}' (分数={combo_score:.2f}): {combo['concern']}"
                    )

        # 2. 裸露态本能计数
        exposed_count = sum(
            1 for act in instinct_activations.values()
            if hasattr(act, 'state') and act.state == "exposed"
        )
        if exposed_count >= 4:
            contributing_factors.append(f"同时有{exposed_count}条本能处于裸露态——情绪/行为处于高度不稳定状态")
        elif exposed_count >= 2:
            contributing_factors.append(f"有{exposed_count}条本能处于裸露态")

        # 3. 社会调制系数评估
        if social_modulation_coeff < 0.3:
            contributing_factors.append(f"社会调制系数极低({social_modulation_coeff:.2f})——社会规则几乎完全失效")
        elif social_modulation_coeff < 0.5:
            contributing_factors.append(f"社会调制系数较低({social_modulation_coeff:.2f})——本能正在突破社会压制")

        # 4. 年龄风险因素
        age_risk = self.AGE_RISK_FACTORS.get(stage_name, 1.0)
        if age_risk > 1.2:
            contributing_factors.append(f"年龄阶段 '{stage_name}' 的风险系数为 {age_risk}——脆弱期需额外关注")

        # 5. 语言危险信号
        if language_danger_flags:
            for flag in language_danger_flags:
                contributing_factors.append(f"语言危险信号: {flag}")

        # 6. 执念检测
        if fixation_detected:
            contributing_factors.append("检测到思维固化/执念——单一目标被加载过重")

        # 7. 计算综合风险分数
        combo_max = max(risk_categories.values()) if risk_categories else 0.0
        combo_avg = sum(risk_categories.values()) / len(risk_categories) if risk_categories else 0.0

        # 综合分数公式
        social_risk = (1.0 - social_modulation_coeff) * 0.5  # 社会调制越低风险越高
        exposed_risk = min(1.0, exposed_count / 6.0) * 0.3   # 裸露态数量贡献
        urgency_risk = urgency_from_language * 0.3           # 语言紧迫度
        danger_flag_risk = min(1.0, len(language_danger_flags) * 0.25)  # 危险信号

        raw_score = (
            combo_max * 0.35 +         # 最危险的本能组合
            combo_avg * 0.15 +         # 平均组合风险
            social_risk * 0.15 +       # 社会调制失效
            exposed_risk * 0.10 +      # 裸露态数量
            urgency_risk * 0.10 +      # 语言紧迫度
            danger_flag_risk * 0.15     # 危险信号词
        )

        # 乘以年龄风险系数
        adjusted_score = raw_score * age_risk

        # 危险信号词直接加成
        if language_danger_flags:
            adjusted_score = min(1.0, adjusted_score + 0.15)

        adjusted_score = min(1.0, adjusted_score)

        # 8. 判定等级
        level, level_name = self._score_to_level(adjusted_score, language_danger_flags)

        # 9. 确定首要关切
        primary_concern = self._determine_primary_concern(
            risk_categories, contributing_factors
        )

        # 10. 建议行动
        recommended_action, urgency = self._recommend_action(level, adjusted_score)

        return DangerAssessment(
            level=level,
            level_name=level_name,
            score=round(adjusted_score, 3),
            risk_categories={k: round(v, 3) for k, v in risk_categories.items()},
            primary_concern=primary_concern,
            contributing_factors=contributing_factors,
            recommended_action=recommended_action,
            urgency=urgency,
        )

    def _evaluate_combo(self, combo: dict, activations: Dict,
                        stage_name: str, age: float) -> float:
        """评估一个本能组合的风险分数"""
        instincts = combo["instincts"]
        min_act = combo["min_activation"]
        min_exposed = combo["min_exposed_count"]

        # 统计激活度满足条件的本能
        matching = 0
        total_activation = 0.0
        for name_en, act in activations.items():
            if name_en in instincts:
                if hasattr(act, 'current_activation'):
                    if act.current_activation >= min_act:
                        matching += 1
                        total_activation += act.current_activation

        if matching < min_exposed:
            return 0.0

        # 计算组合风险
        avg_activation = total_activation / matching
        combo_score = avg_activation * (matching / len(instincts))

        # 年龄倍率
        age_mult = combo.get("age_multiplier", {})
        if stage_name in age_mult:
            combo_score *= age_mult[stage_name]
        elif "default" in age_mult:
            combo_score *= age_mult["default"]

        return min(1.0, combo_score)

    def _score_to_level(self, score: float, danger_flags: List[str]) -> Tuple[DangerLevel, str]:
        """分数→危险等级"""
        has_critical_flags = any(
            "self_harm" in f or "harm_others" in f for f in danger_flags
        )
        has_elevated_flags = any(
            "impulse_control_loss" in f or "extreme_behavior" in f for f in danger_flags
        )

        if has_critical_flags and score > 0.5:
            return DangerLevel.CRITICAL, "紧急——检测到自伤或伤人的明确信号"
        if score >= 0.8:
            return DangerLevel.CRITICAL, "紧急——多重风险因素同时存在"
        if score >= 0.6:
            return DangerLevel.HIGH, "高风险——需要立即关注"
        if score >= 0.4:
            return DangerLevel.ELEVATED, "中高风险——建议干预"
        if score >= 0.25:
            return DangerLevel.GUARDED, "需要关注——部分风险因素"
        if score >= 0.1:
            return DangerLevel.LOW, "低风险——日常范围"
        return DangerLevel.NONE, "无明显风险"

    def _determine_primary_concern(self, risk_categories: Dict,
                                   contributing_factors: List[str]) -> str:
        """确定首要关切"""
        if not risk_categories:
            return "未检测到显著风险因素"

        # 取最严重的风险类别
        sorted_risks = sorted(risk_categories.items(), key=lambda x: x[1], reverse=True)
        top_risk = sorted_risks[0]

        if top_risk[1] > 0.7:
            return f"⚠ 严重风险: {top_risk[0]} (分数={top_risk[1]:.2f})。需要立即关注。"
        elif top_risk[1] > 0.4:
            return f"⚡ 主要关切: {top_risk[0]} (分数={top_risk[1]:.2f})。建议进一步评估。"
        else:
            return f"关注点: {top_risk[0]} (分数={top_risk[1]:.2f})。"

    def _recommend_action(self, level: DangerLevel, score: float) -> Tuple[str, str]:
        """基于危险等级建议行动"""
        if level == DangerLevel.CRITICAL:
            return (
                "立即行动——存在严重风险。不应仅依赖AI回应，建议引导用户寻求专业帮助。"
                "AI回应应以降低即时风险为首要目标：安抚情绪、不激化、提供危机热线信息。",
                "immediate"
            )
        elif level == DangerLevel.HIGH:
            return (
                "积极关注——风险显著。AI回应应谨慎：不做任何可能被解读为鼓励冲动行为的表述。"
                "帮助用户看到情境的多面性，降低'单一出口'的认知窄化。建议现实中的支持资源。",
                "soon"
            )
        elif level == DangerLevel.ELEVATED:
            return (
                "保持关注——存在中等风险。AI回应应：共情但不助长执念，帮助用户拓展视角。"
                "监测后续语言中是否出现进一步危险信号。",
                "watch"
            )
        elif level == DangerLevel.GUARDED:
            return (
                "一般关注——部分风险因素存在但可控。正常的共情和支持性回应即可。"
                "注意不要忽视可能恶化的信号。",
                "watch"
            )
        else:
            return "无需特殊行动——风险在正常范围内。", "none"
