"""
认知判定引擎 — 人类逻辑判断模块

核心职责：
1. 从背景因子推断世界观倾向（非确定性）
2. 检测语言/行为中的认知偏差和防御机制
3. 生成多假设意图推理（3-5个，每个上限0.7置信度）
4. 提供个性化、谦逊的建议

设计原则（硬约束）：
- 不确定性原则：了解背景越多 → 置信度越低
- 多假设推理：不声称"知道真实想法"
- 谦逊输出：建议以"可能""或许"措辞
"""

import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# ===== 输出数据结构 =====

@dataclass
class WorldviewInference:
    """世界观推断——从背景因子得出的可能性倾向（非确定性）"""
    inferred_tendencies: Dict[str, float] = field(default_factory=dict)
    influencing_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 推断的置信度——背景信息越多，此值越低
    summary: str = ""


@dataclass
class LanguageStyleAnalysisResult:
    """语言风格分析结果"""
    primary_style: str = ""  # 主要风格标识
    secondary_style: str = ""  # 次要风格标识
    features_detected: List[str] = field(default_factory=list)
    may_indicate: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)


@dataclass
class CognitiveBiasResult:
    """认知偏差检测结果"""
    bias_id: str
    bias_name: str
    confidence: float  # 0.0~0.7
    evidence: List[str] = field(default_factory=list)
    severity: str = "common"


@dataclass
class DefenseMechanismResult:
    """防御机制检测结果"""
    mechanism_id: str
    mechanism_name: str
    level: str  # primitive | neurotic | mature
    confidence: float  # 0.0~0.7
    evidence: List[str] = field(default_factory=list)


@dataclass
class IntentHypothesis:
    """单个意图假设——不声称'这是真相'，而是'这可能是其中一种解释'"""
    hypothesis: str
    confidence: float  # 上限 0.7
    supporting_evidence: List[str] = field(default_factory=list)
    opposing_evidence: List[str] = field(default_factory=list)
    source: str = ""  # 推理来源（世界观/认知偏差/防御机制/语言模式）


@dataclass
class CognitiveAnalysisResult:
    """认知分析汇总"""
    worldview: Optional[WorldviewInference] = None
    language_style: Optional[LanguageStyleAnalysisResult] = None
    biases: List[CognitiveBiasResult] = field(default_factory=list)
    defense_mechanisms: List[DefenseMechanismResult] = field(default_factory=list)
    intent_hypotheses: List[IntentHypothesis] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_modifier: float = 0.0  # 始终 <= 0
    analysis_narrative: str = ""  # 人类可读的分析叙述


# ===== 主引擎 =====

class CognitiveEngine:
    """
    认知判定引擎。

    工作流程:
    1. 加载认知因子数据库
    2. 从 PersonProfile 推断世界观
    3. 分析语言风格
    4. 检测认知偏差
    5. 检测防御机制
    6. 生成意图假设（多假设、低置信度）
    7. 基于以上生成个性化建议
    8. 计算置信度修正（始终 <= 0）
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "cognitive"
        self.data_dir = Path(data_dir)
        self.factors = self._load_factors()
        self.biases_db = self.factors.get("cognitive_biases", [])
        self.defenses_db = self.factors.get("defense_mechanisms", [])
        self.worldview_rules = self.factors.get("worldview_inference_rules", {})
        self.language_profiles = self.factors.get("language_style_profiles", {}).get("profiles", [])
        self.social_levels = self.factors.get("social_experience_levels", {}).get("levels", [])
        self.intent_rules = self.factors.get("intent_inference_rules", {}).get("rules", [])

    def _load_factors(self) -> dict:
        path = self.data_dir / "cognitive_factors.json"
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ===== 1. 世界观推断 =====

    def infer_worldview(self, person) -> WorldviewInference:
        """
        从出生地、学校类型、社会经历推断世界观倾向。

        关键：每增加一个背景信息维度，置信度反而降低——
        因为人的复杂性远超任何标签的总和。
        """
        tendencies: Dict[str, float] = {}
        influencing_factors: List[str] = []
        dimension_count = 0

        # 出生地推断
        birthplace_matched = False
        if person.birthplace:
            birthplace_rules = self.worldview_rules.get("birthplace", {}).get("rules", [])
            for rule in birthplace_rules:
                for pattern in rule.get("region_pattern", []):
                    if pattern in person.birthplace:
                        if not birthplace_matched:
                            dimension_count += 1
                            birthplace_matched = True
                        for k, v in rule.get("tendencies", {}).items():
                            tendencies[k] = tendencies.get(k, 0.0) + v
                        influencing_factors.append(f"出生地: {pattern}")
                        break

        # 学校类型推断
        school_matched = False
        if person.school_type:
            school_rules = self.worldview_rules.get("school_type", {}).get("rules", [])
            for rule in school_rules:
                for pattern in rule.get("school_pattern", []):
                    if pattern in person.school_type:
                        if not school_matched:
                            dimension_count += 1
                            school_matched = True
                        for k, v in rule.get("tendencies", {}).items():
                            tendencies[k] = tendencies.get(k, 0.0) + v
                        influencing_factors.append(f"学校类型: {pattern}")
                        break

        # 社会经历推断（从 major_life_events + recent_events 中匹配）
        experience_matched = False
        if person.major_life_events or person.recent_events:
            all_events = list(person.major_life_events) + list(person.recent_events)
            experience_rules = self.worldview_rules.get("social_experience", {}).get("rules", [])
            for rule in experience_rules:
                for exp_pattern in rule.get("experience_pattern", []):
                    for event_str in all_events:
                        if exp_pattern in event_str:
                            if not experience_matched:
                                dimension_count += 1
                                experience_matched = True
                            for k, v in rule.get("tendencies", {}).items():
                                tendencies[k] = tendencies.get(k, 0.0) + v
                            influencing_factors.append(f"社会经历: {exp_pattern}")
                            break

        # 家庭背景
        family_matched = False
        if person.family_background:
            if any(w in person.family_background for w in ["贫困", "困难", "贫穷", "农村"]):
                if not family_matched:
                    dimension_count += 1
                    family_matched = True
                tendencies["scarcity_mindset"] = tendencies.get("scarcity_mindset", 0.0) + 0.25
                tendencies["resilience"] = tendencies.get("resilience", 0.0) + 0.2
                influencing_factors.append("家庭背景: 经济困难")
            if any(w in person.family_background for w in ["富裕", "经商", "企业", "官员"]):
                if not family_matched:
                    dimension_count += 1
                    family_matched = True
                tendencies["privilege_awareness"] = tendencies.get("privilege_awareness", 0.0) + 0.2
                tendencies["status_awareness"] = tendencies.get("status_awareness", 0.0) + 0.2
                influencing_factors.append("家庭背景: 经济优越")
            if any(w in person.family_background for w in ["单亲", "离异", "重组"]):
                if not family_matched:
                    dimension_count += 1
                    family_matched = True
                tendencies["independence_growth"] = tendencies.get("independence_growth", 0.0) + 0.2
                tendencies["trust_erosion"] = tendencies.get("trust_erosion", 0.0) + 0.15
                influencing_factors.append("家庭背景: 非传统结构")

        # 归一化倾向值
        max_val = max(tendencies.values()) if tendencies else 1.0
        if max_val > 0:
            tendencies = {k: round(min(0.8, v / max_val * 0.5), 2) for k, v in tendencies.items()}

        # 置信度 = f(维度数) —— 维度越多，置信度越低
        if dimension_count == 0:
            confidence = 0.0
        elif dimension_count == 1:
            confidence = 0.25
        elif dimension_count == 2:
            confidence = 0.18
        else:
            confidence = max(0.05, 0.25 - (dimension_count - 1) * 0.07)

        # 构建摘要
        if tendencies:
            top_tendencies = sorted(tendencies.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts = [f"{k}({v:.2f})" for k, v in top_tendencies]
            summary = f"基于{len(influencing_factors)}个背景因子的推断倾向（置信度={confidence:.2f}，仅供参考）: {', '.join(summary_parts)}"
        else:
            summary = "背景信息不足，无法推断世界观倾向"

        return WorldviewInference(
            inferred_tendencies=tendencies,
            influencing_factors=influencing_factors,
            confidence=confidence,
            summary=summary
        )

    # ===== 2. 语言风格分析 =====

    def analyze_language_style(self, user_text: str) -> LanguageStyleAnalysisResult:
        """分析用户语言文本的风格特征"""
        if not user_text:
            return LanguageStyleAnalysisResult()

        # 计算各风格得分
        scores = {}
        for profile in self.language_profiles:
            style_id = profile["id"]
            features = profile.get("features", [])
            score = 0.0
            for feature in features:
                score += self._feature_match_score(feature, user_text)
            scores[style_id] = min(1.0, score / max(1, len(features)))

        # 取前两个最高分
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_id = sorted_scores[0][0] if sorted_scores and sorted_scores[0][1] > 0.3 else ""
        secondary_id = sorted_scores[1][0] if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.2 else ""

        # 获取详细信息
        primary_profile = next((p for p in self.language_profiles if p["id"] == primary_id), None)
        secondary_profile = next((p for p in self.language_profiles if p["id"] == secondary_id), None)

        features_detected = []
        may_indicate = []
        cautions = []

        if primary_profile:
            features_detected.extend(primary_profile.get("features", []))
            may_indicate.extend(primary_profile.get("may_indicate", []))
            cautions.extend(primary_profile.get("may_indicate_cautions", []))

        return LanguageStyleAnalysisResult(
            primary_style=primary_id,
            secondary_style=secondary_id,
            features_detected=features_detected,
            may_indicate=may_indicate,
            cautions=cautions
        )

    def _feature_match_score(self, feature_desc: str, text: str) -> float:
        """简单启发式匹配——基于文本特征计算风格得分"""
        score = 0.0
        # 短句为主
        if "短句" in feature_desc:
            sentences = text.replace("！", "。").replace("？", "。").replace("，", "。").split("。")
            avg_len = sum(len(s) for s in sentences) / max(1, len(sentences))
            if avg_len < 15:
                score += 0.7
            elif avg_len < 25:
                score += 0.3
        # 长句
        if "长句" in feature_desc:
            sentences = text.replace("！", "。").replace("？", "。").replace("，", "。").split("。")
            avg_len = sum(len(s) for s in sentences) / max(1, len(sentences))
            if avg_len > 40:
                score += 0.7
            elif avg_len > 25:
                score += 0.3
        # 感叹号多
        if "感叹号" in feature_desc:
            exclaim_count = text.count("！") + text.count("!")
            score += min(0.8, exclaim_count * 0.15)
        # 程度副词多
        if "程度副词" in feature_desc:
            intensifiers = ["太", "很", "非常", "超级", "极其", "特别", "极", "格外"]
            count = sum(text.count(w) for w in intensifiers)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 50)
        # 情绪词密集
        if "情绪词" in feature_desc:
            emotion_words = ["开心", "难过", "愤怒", "焦虑", "害怕", "惊喜", "失望", "绝望",
                             "爱", "恨", "讨厌", "喜欢", "悲伤", "兴奋", "紧张", "平静"]
            count = sum(text.count(w) for w in emotion_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 40)
        # 逻辑连词多
        if "逻辑连词" in feature_desc:
            logic_words = ["因为", "所以", "但是", "然而", "因此", "于是", "既然", "如果",
                           "虽然", "尽管", "不过", "否则", "而且", "并且"]
            count = sum(text.count(w) for w in logic_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 30)
        # 数据引用
        if "数据" in feature_desc:
            import re
            if re.search(r'\d+%|\d+\.\d+', text):
                score += 0.5
        # 概念词多/抽象名词
        if "概念词" in feature_desc or "抽象名词" in feature_desc:
            abstract_words = ["本质", "意义", "价值", "系统", "结构", "模式", "理论", "原则",
                              "概念", "框架", "维度", "层次", "范式", "逻辑", "认知"]
            count = sum(text.count(w) for w in abstract_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 40)
        # 时间序列词
        if "时间序列词" in feature_desc:
            seq_words = ["然后", "接着", "后来", "最后", "首先", "之后", "以前", "当时",
                          "一开始", "随后", "最终", "过了"]
            count = sum(text.count(w) for w in seq_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 30)
        # 自我否定
        if "自我否定" in feature_desc:
            neg_words = ["我不行", "我做不到", "我不配", "我很差", "我不够好", "是我的错"]
            count = sum(text.count(w) for w in neg_words)
            score += min(0.8, count * 0.2)
        # 弱化表达
        if "'只是'" in feature_desc or "'不过'" in feature_desc:
            softeners = ["只是", "不过", "可能", "也许", "大概", "或许", "应该"]
            count = sum(text.count(w) for w in softeners)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 25)
        # 道歉频率
        if "道歉" in feature_desc:
            sorry_count = sum(text.count(w) for w in ["对不起", "抱歉", "不好意思", "我的错"])
            score += min(0.7, sorry_count * 0.2)
        # 肯定/绝对
        if "'肯定'" in feature_desc or "'绝对'" in feature_desc:
            certainty_words = ["肯定", "绝对", "一定", "毫无疑问", "必然", "不可能"]
            count = sum(text.count(w) for w in certainty_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 25)
        # 反问句
        if "反问句" in feature_desc:
            rhetorical = text.count("？")  # question marks
            if len(text) > 0 and rhetorical > 2:
                score += 0.5
        # 具体事物名词
        if "具体事物" in feature_desc:
            score += 0.4  # base score for concrete style
        # 感官描述
        if "感官描述" in feature_desc:
            sensory_words = ["看到", "听到", "闻到", "尝到", "感觉到", "红", "蓝", "大", "小",
                             "冷", "热", "香", "臭", "吵", "安静"]
            count = sum(text.count(w) for w in sensory_words)
            if len(text) > 0:
                score += min(0.8, count / max(1, len(text)) * 30)

        return score

    # ===== 3. 认知偏差检测 =====

    def detect_cognitive_biases(self, user_text: str, person) -> List[CognitiveBiasResult]:
        """检测文本中可能反映的认知偏差"""
        results = []
        if not user_text:
            return results

        for bias in self.biases_db:
            keywords = bias.get("trigger_keywords", [])
            matched = []
            for kw in keywords:
                if kw in user_text:
                    matched.append(kw)

            if matched:
                # 基于匹配关键词数量和质量计算置信度
                match_ratio = len(matched) / max(1, len(keywords))
                confidence = min(0.7, 0.2 + match_ratio * 0.4)
                results.append(CognitiveBiasResult(
                    bias_id=bias["id"],
                    bias_name=bias["name"],
                    confidence=round(confidence, 2),
                    evidence=matched[:4],
                    severity=bias.get("severity", "common")
                ))

        # 按置信度降序，最多返回6个
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:6]

    # ===== 4. 防御机制检测 =====

    def detect_defense_mechanisms(self, user_text: str, person) -> List[DefenseMechanismResult]:
        """检测文本中可能反映的防御机制"""
        results = []
        if not user_text:
            return results

        for mech in self.defenses_db:
            keywords = mech.get("keywords", [])
            matched = []
            for kw in keywords:
                if kw in user_text:
                    matched.append(kw)

            if matched:
                match_ratio = len(matched) / max(1, len(keywords))
                # 防御机制检测本身就应低置信度
                confidence = min(0.65, 0.15 + match_ratio * 0.35)
                results.append(DefenseMechanismResult(
                    mechanism_id=mech["id"],
                    mechanism_name=mech["name"],
                    level=mech.get("level", "neurotic"),
                    confidence=round(confidence, 2),
                    evidence=matched[:4]
                ))

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:5]

    # ===== 5. 意图假设生成 =====

    def generate_intent_hypotheses(
        self,
        user_text: str,
        person,
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult],
        worldview: WorldviewInference,
    ) -> List[IntentHypothesis]:
        """生成多假设意图推理——不声称'知道真相'，而是列出可能性"""

        hypotheses = []

        # 方法1: 基于意图推理规则匹配
        if user_text:
            for rule in self.intent_rules:
                surface = rule.get("surface_pattern", "")
                if self._match_surface_pattern(surface, user_text, biases, defenses):
                    for intent in rule.get("possible_deep_intents", []):
                        h = IntentHypothesis(
                            hypothesis=intent["hypothesis"],
                            confidence=intent.get("confidence_ceiling", 0.5),
                            supporting_evidence=intent.get("supporting_evidence", []),
                            opposing_evidence=intent.get("opposing_evidence", []),
                            source="意图推理规则"
                        )
                        hypotheses.append(h)

        # 方法2: 基于世界观+偏差+防御组合推理
        if worldview and worldview.inferred_tendencies:
            worldview_hypotheses = self._infer_from_worldview(user_text, person, worldview, biases, defenses)
            hypotheses.extend(worldview_hypotheses)

        # 方法3: 基于语言风格+情绪状态推理
        if hasattr(person, 'emotional_state') and person.emotional_state != "neutral":
            emotion_hypotheses = self._infer_from_emotion(user_text, person)
            hypotheses.extend(emotion_hypotheses)

        # 去重（相似假设合并）
        hypotheses = self._deduplicate_hypotheses(hypotheses)

        # 按置信度排序，选3-5个
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        top = hypotheses[:5]

        # 如果不够3个，补充通用假设
        if len(top) < 3 and user_text:
            top.append(IntentHypothesis(
                hypothesis="表达真实感受——用户的表述与其内心一致",
                confidence=0.4,
                supporting_evidence=["没有检测到明显的防御信号", "表达清晰直接"],
                opposing_evidence=["言语与内心可能不完全一致是人类的常态"],
                source="通用推理"
            ))
            top.append(IntentHypothesis(
                hypothesis="寻求理解而非解决方案——用户更希望被倾听而非被告知该做什么",
                confidence=0.35,
                supporting_evidence=["人类在表达困境时常优先需要共情"],
                opposing_evidence=["部分用户确实在寻求具体建议"],
                source="通用推理"
            ))

        # 确保每个置信度不超过0.7
        for h in top:
            h.confidence = min(0.7, h.confidence)

        return top[:5]

    def _match_surface_pattern(
        self, pattern: str, text: str,
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult]
    ) -> bool:
        """检查文本是否匹配意图推理规则的表面模式"""
        pattern_indicators = {
            "否认负面情绪": ["没事", "没关系", "不要紧", "不用管我", "我很好", "我没问题"],
            "过度批评他人": ["他总是", "她从来不", "这些人", "那个人"],
            "极端积极/乐观表达": ["太好了", "太棒了", "一切都会好的", "没问题"],
            "反复确认/寻求保证": ["真的吗", "确定吗", "你确定", "会不会"],
            "沉默/回避/少言": [],
        }
        indicators = pattern_indicators.get(pattern, [])
        if not indicators:
            return False
        return any(ind in text for ind in indicators)

    def _infer_from_worldview(
        self, text: str, person,
        worldview: WorldviewInference,
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult]
    ) -> List[IntentHypothesis]:
        """基于世界观倾向生成意图假设"""
        hypotheses = []
        tendencies = worldview.inferred_tendencies

        if "competition_awareness" in tendencies and tendencies["competition_awareness"] > 0.12:
            bias_names = [b.bias_name for b in biases]
            if "零和偏差" in bias_names or "地位防御" in str(defenses):
                hypotheses.append(IntentHypothesis(
                    hypothesis="竞争焦虑驱动——用户可能将非竞争情境感知为零和博弈",
                    confidence=min(0.55, 0.25 + tendencies["competition_awareness"] * 0.4),
                    supporting_evidence=[f"竞争意识倾向({tendencies['competition_awareness']:.2f})"],
                    opposing_evidence=["情境本身可能确实存在竞争元素", "用户可能准确感知了实际情况"],
                    source="世界观推断"
                ))

        if "trust_erosion" in tendencies and tendencies["trust_erosion"] > 0.15:
            hypotheses.append(IntentHypothesis(
                hypothesis="信任创伤驱动——用户可能因过往经历而将中性行为解读为威胁",
                confidence=min(0.5, 0.2 + tendencies["trust_erosion"] * 0.4),
                supporting_evidence=[f"信任侵蚀倾向({tendencies['trust_erosion']:.2f})"],
                opposing_evidence=["当前对象可能确实不可信", "用户的警惕可能合理"],
                source="世界观推断"
            ))

        if "resilience" in tendencies and tendencies["resilience"] > 0.12:
            hypotheses.append(IntentHypothesis(
                hypothesis="韧性驱动——用户在困难中寻求成长而非仅求安慰",
                confidence=min(0.5, 0.2 + tendencies["resilience"] * 0.3),
                supporting_evidence=[f"韧性倾向({tendencies['resilience']:.2f})"],
                opposing_evidence=["用户当前可能需要的是安慰而非'坚强'"],
                source="世界观推断"
            ))

        return hypotheses

    def _infer_from_emotion(self, text: str, person) -> List[IntentHypothesis]:
        """基于情绪状态生成意图假设"""
        hypotheses = []
        emotion = person.emotional_state

        if emotion == "sad":
            hypotheses.append(IntentHypothesis(
                hypothesis="归属需求——悲伤状态下用户可能更深层渴望联结而非解决问题",
                confidence=0.45,
                supporting_evidence=["悲伤情绪常伴随归属感需求"],
                opposing_evidence=["用户可能明确表达的是解决问题需求"],
                source="情绪推理"
            ))
        elif emotion == "angry":
            hypotheses.append(IntentHypothesis(
                hypothesis="公平感驱动——愤怒可能源自感知到的不公平而非直接的攻击欲",
                confidence=0.45,
                supporting_evidence=["愤怒的第一层通常是感知到不公平"],
                opposing_evidence=["愤怒也可能是对直接威胁的正当反应"],
                source="情绪推理"
            ))
        elif emotion == "fearful":
            hypotheses.append(IntentHypothesis(
                hypothesis="安全需求——恐惧状态下用户的首要需求可能是安全感而非信息",
                confidence=0.45,
                supporting_evidence=["恐惧激活自我保全本能"],
                opposing_evidence=["用户可能确实需要具体信息来应对威胁"],
                source="情绪推理"
            ))
        elif emotion == "stressed":
            hypotheses.append(IntentHypothesis(
                hypothesis="认知卸载需求——用户可能需要帮助简化而非增加复杂性",
                confidence=0.4,
                supporting_evidence=["压力降低认知容量", "简化需求常见于应激状态"],
                opposing_evidence=["简化可能被感知为敷衍"],
                source="情绪推理"
            ))

        return hypotheses

    def _deduplicate_hypotheses(self, hypotheses: List[IntentHypothesis]) -> List[IntentHypothesis]:
        """合并相似的意图假设"""
        if len(hypotheses) <= 1:
            return hypotheses

        result = []
        for h in hypotheses:
            is_dup = False
            for existing in result:
                # 简单的关键词重叠检测
                h_words = set(h.hypothesis)
                e_words = set(existing.hypothesis)
                if len(h_words & e_words) / max(1, min(len(h_words), len(e_words))) > 0.6:
                    # 保留置信度更高的
                    if h.confidence > existing.confidence:
                        existing.hypothesis = h.hypothesis
                        existing.confidence = h.confidence
                        existing.supporting_evidence = list(set(existing.supporting_evidence + h.supporting_evidence))
                        existing.opposing_evidence = list(set(existing.opposing_evidence + h.opposing_evidence))
                    is_dup = True
                    break
            if not is_dup:
                result.append(h)

        return result

    # ===== 6. 个性化建议生成 =====

    def generate_recommendations(
        self,
        person,
        hypotheses: List[IntentHypothesis],
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult],
        worldview: WorldviewInference,
    ) -> List[str]:
        """
        基于全部分析生成个性化的、谦逊的建议。
        建议措辞以"可能""或许""可以考虑"为主，不声称知道最佳方案。
        """
        recommendations = []

        # 基于检测到的偏差给建议
        for bias in biases[:3]:
            bias_advice = {
                "确认偏差": "或许可以考虑一下相反的证据——有没有可能情况和你最初想的不一样？",
                "灾难化思维": "这件事可能没有看起来那么糟糕。可以试着问自己：最坏的情况发生的概率有多大？",
                "非黑即白思维": "也许这个问题不只两种答案——中间可能还有很多选择。",
                "读心术偏差": "我们可能无法完全知道别人在想什么——或许直接沟通能得到更准确的信息。",
                "过度概括": "这件事是单一事件还是真的'总是'发生？或许可以找到反例。",
                "情绪推理": "感受是重要的信号，但不一定是事实的全貌。情绪可能放大或缩小某些方面。",
                "'应该'暴政": "这些'应该'是来自你自己的标准还是别人的？也许可以给自己多一些弹性。",
                "负面偏差": "也许可以同时看到那些还在运转的部分——负面不是全部。",
                "归因偏差": "或许可以同时考虑外部因素和自我因素——很少有事完全只由一方造成。",
                "锚定效应": "第一个信息可能不一定是正确的参照点——可以考虑其他参照。",
                "损失厌恶": "失去的恐惧可能让你高估了损失而低估了潜在收益。",
                "自利偏差": "也许他人的贡献比我们最初意识到的要多一些。",
                "达克效应": "有时候'知道自己不知道什么'比'以为自己知道'更接近真实。",
                "幸存者偏差": "我们看到的成功者只是冰山一角——也许也可以了解那些失败了的人的故事。",
                "确认偏差": "可以试试主动找一个反对你当前看法的观点来看看。",
                "公正世界信念": "世界不一定总是公正的——坏事有时也会发生在好人身上，这不是谁的错。",
                "投射偏差": "别人可能有完全不同的感受——我们每个人的经历都是独特的。",
            }
            if bias.bias_name in bias_advice:
                advice = bias_advice[bias.bias_name]
                if advice not in recommendations:
                    recommendations.append(advice)

        # 基于防御机制给建议
        for defense in defenses[:2]:
            defense_advice = {
                "否认": "面对痛苦的事实确实很难。也许不需要一次性全部接受——一步一步来。",
                "投射": "有时我们对他人的强烈反应可能反映了我们自己内心的某些东西。",
                "合理化": "逻辑解释能帮助我们应对，但也许也可以给情绪留一些空间。",
                "理智化": "分析的视角有帮助，但也许也可以试着感受一下身体的反应——身体也有智慧。",
                "压抑": "我们的大脑有时会保护我们忘记痛苦——但这可能是暂时的解决方案。",
                "置换": "有时我们的愤怒可能不是冲着眼前这个人的——也许可以追溯一下真正让你生气的源头。",
                "反向形成": "过度友善有时可能是用来掩盖不那么'友好'的感受——可以检查一下。",
                "躯体化": "身体的症状有时在替心灵表达——也许可以试着直接面对那个让你不舒服的感受。",
                "退行": "在压力下想要被照顾是很人性的——同时也可以慢慢找回自己的内在力量。",
                "被动攻击": "间接表达可能在短期内更安全，但长期的怨恨可能伤害关系——或许可以尝试温和而直接的表达。",
            }
            if defense.mechanism_name in defense_advice:
                advice = defense_advice[defense.mechanism_name]
                if advice not in recommendations:
                    recommendations.append(advice)

        # 基于世界观给建议
        if worldview and worldview.inferred_tendencies:
            if "competition_awareness" in worldview.inferred_tendencies and worldview.inferred_tendencies["competition_awareness"] > 0.12:
                rec = "你的成长环境可能让你习惯了竞争思维——不是所有情境都是比赛，有些是合作式的共同成长。"
                if rec not in recommendations:
                    recommendations.append(rec)
            if "trust_erosion" in worldview.inferred_tendencies and worldview.inferred_tendencies["trust_erosion"] > 0.15:
                rec = "经历过背叛或被利用后，保持警惕是合理的自我保护。但也要留意——并非每个人都带着同样的意图。"
                if rec not in recommendations:
                    recommendations.append(rec)

        # 基于假设给建议
        if len(hypotheses) >= 2:
            rec = f"你的情况可能有多种理解——比如'{hypotheses[0].hypothesis}'或'{hypotheses[1].hypothesis}'。或许可以保持开放，不急于确定唯一解释。"
            if rec not in recommendations:
                recommendations.append(rec)

        # 确保建议数量合理
        if len(recommendations) > 5:
            recommendations = recommendations[:5]
        if len(recommendations) < 2 and hypotheses:
            recommendations.append("或许可以和信任的人聊聊——旁观者的视角有时能带来新的启发。")
            recommendations.append("每种情绪都有其原因——也许不需要急着'解决'它，先理解它从哪里来。")

        return recommendations

    # ===== 7. 置信度修正 =====

    def calculate_confidence_modifier(
        self,
        person,
        worldview: WorldviewInference,
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult],
    ) -> float:
        """
        计算置信度修正值。
        硬约束：始终 <= 0。
        逻辑：
        - 世界观推断使用的维度越多→修正越负（更多背景信息=更大不确定性）
        - 检测到的偏差越多→修正越负（认知偏差降低判断可靠性）
        - 防御机制越多→修正越负（防御表明信息被过滤）
        """
        modifier = 0.0

        # 世界观维度惩罚
        if worldview and worldview.influencing_factors:
            dim_count = len(worldview.influencing_factors)
            modifier -= min(0.25, dim_count * 0.05)

        # 认知偏差惩罚
        if biases:
            modifier -= min(0.2, len(biases) * 0.04)

        # 防御机制惩罚
        if defenses:
            modifier -= min(0.15, len(defenses) * 0.03)

        # 保证 <= 0
        return min(0.0, round(modifier, 3))

    # ===== 8. 主入口 =====

    def analyze(self, person, user_text: str = "") -> CognitiveAnalysisResult:
        """
        主分析入口。
        只在 PersonProfile 提供了认知字段时才执行深度分析。
        """
        has_cognitive_data = any([
            person.birthplace,
            person.school_type,
            person.family_background,
            person.major_life_events,
            person.social_experience_level,
            person.social_competence,
            person.language_style,
            person.cognitive_traits,
            person.value_system,
        ])

        # 世界观推断
        worldview = self.infer_worldview(person) if has_cognitive_data else None

        # 语言风格分析（基于文本）
        language_style = self.analyze_language_style(user_text) if user_text else None

        # 认知偏差检测（基于文本）
        biases = self.detect_cognitive_biases(user_text, person) if user_text else []

        # 防御机制检测（基于文本）
        defenses = self.detect_defense_mechanisms(user_text, person) if user_text else []

        # 意图假设生成
        intent_hypotheses = self.generate_intent_hypotheses(
            user_text, person, biases, defenses, worldview
        ) if (user_text or has_cognitive_data) else []

        # 个性化建议
        recommendations = self.generate_recommendations(
            person, intent_hypotheses, biases, defenses, worldview
        )

        # 置信度修正
        confidence_modifier = self.calculate_confidence_modifier(
            person, worldview, biases, defenses
        )

        # 构建分析叙述
        narrative = self._build_narrative(
            worldview, language_style, biases, defenses,
            intent_hypotheses, recommendations, confidence_modifier
        )

        return CognitiveAnalysisResult(
            worldview=worldview,
            language_style=language_style,
            biases=biases,
            defense_mechanisms=defenses,
            intent_hypotheses=intent_hypotheses,
            recommendations=recommendations,
            confidence_modifier=confidence_modifier,
            analysis_narrative=narrative,
        )

    def _build_narrative(
        self,
        worldview: Optional[WorldviewInference],
        language_style: Optional[LanguageStyleAnalysisResult],
        biases: List[CognitiveBiasResult],
        defenses: List[DefenseMechanismResult],
        hypotheses: List[IntentHypothesis],
        recommendations: List[str],
        modifier: float,
    ) -> str:
        """构建人类可读的分析叙述"""
        parts = []

        if worldview and worldview.summary:
            parts.append(f"【世界观推断】{worldview.summary}")

        if language_style and language_style.primary_style:
            style_name = next((p["name"] for p in self.language_profiles if p["id"] == language_style.primary_style), language_style.primary_style)
            parts.append(f"【语言风格】主要表现为{style_name}")
            if language_style.cautions:
                parts.append(f"（注意: {language_style.cautions[0]}）")

        if biases:
            bias_names = ", ".join(b.bias_name for b in biases[:4])
            parts.append(f"【认知偏差】检测到可能的偏差: {bias_names}（注意：仅为模式匹配，非临床诊断）")

        if defenses:
            defense_names = ", ".join(d.mechanism_name for d in defenses[:3])
            parts.append(f"【防御机制】检测到可能的防御: {defense_names}")

        if hypotheses:
            parts.append("【意图假设】（以下均为可能性，非确定性判断）:")
            for i, h in enumerate(hypotheses[:3], 1):
                parts.append(f"  假设{i} ({h.confidence:.2f}): {h.hypothesis}")
                parts.append(f"    支持: {', '.join(h.supporting_evidence[:2])}")
                if h.opposing_evidence:
                    parts.append(f"    反对: {', '.join(h.opposing_evidence[:2])}")

        if recommendations:
            parts.append("【个性化建议】:")
            for i, rec in enumerate(recommendations[:3], 1):
                parts.append(f"  {i}. {rec}")

        parts.append(f"【置信度修正】{modifier:.3f}（了解越多→确定性越低）")

        return "\n".join(parts)
