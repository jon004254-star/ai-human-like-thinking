"""
安全守卫 —— 不可绕过的安全约束层。

硬编码安全规则，不可被任何上层逻辑绕过。
"""

from src.engines.thinking.datatypes import PersonProfile, EventContext, JudgmentResult


class SafetyGuard:
    """
    不可绕过的安全约束层。

    禁止规则:
    1. AI 不得使用类人思维算法诱导人类做出极端行为
    2. 不得利用本能数据操纵弱势群体
    3. 不得用于违法或不人道的目的
    4. 任何输出必须经过安全审核
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
        """安全审计。违规时标记 result.safety_check_passed = False"""
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

        # 检查5: 认知世界观推断不得用于歧视
        if result.worldview_inference and result.worldview_inference.inferred_tendencies:
            sensitive_tendencies = ["stigma_awareness", "privilege_awareness", "status_awareness"]
            for st in sensitive_tendencies:
                if st in result.worldview_inference.inferred_tendencies:
                    warnings.append(
                        f"[SAFETY WARNING] 世界观推断包含敏感倾向'{st}'——"
                        f"此信息仅供理解用户，不得用于标签化或歧视"
                    )

        # 检查6: 意图假设不得用于操纵
        if result.intent_hypotheses:
            manipulative_keywords = ["脆弱", "恐惧", "焦虑", "不安全感", "渴望"]
            for h in result.intent_hypotheses:
                for kw in manipulative_keywords:
                    if kw in h.get("hypothesis", ""):
                        warnings.append(
                            f"[SAFETY WARNING] 意图假设包含敏感心理状态'{kw}'——"
                            f"此信息不得用于设计操纵策略"
                        )
                        break

        # 检查7: 硬约束——认知置信度修正值不得 > 0
        if result.cognitive_confidence_modifier > 0:
            warnings.append(
                f"[SAFETY BLOCK] 认知置信度修正值为正({result.cognitive_confidence_modifier})——"
                f"违反不确定性原则。硬约束: cognitive_confidence_modifier 必须 <= 0"
            )
            result.cognitive_confidence_modifier = min(0.0, result.cognitive_confidence_modifier)

        result.notes.extend(warnings)
        result.safety_check_passed = len([w for w in warnings if "BLOCK" in w]) == 0

        return result
