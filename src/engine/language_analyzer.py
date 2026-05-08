"""
语言分析模块

从人类自然语言中提取本能信号。
核心能力：
1. 检测社会过滤——当用户用'安全'的语言表达'不安全'的冲动
2. 委婉语映射——将社会可接受的说法映射到真实本能
3. 语言-情绪-本能的三层关联分析
4. 危险信号词检测
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LanguageSignal:
    """从语言中提取的本能信号"""
    instinct_name_en: str
    instinct_name: str
    confidence: float  # 0.0 ~ 1.0 该语言信号对应该本能的置信度
    evidence: List[str]  # 支撑证据（原文片段）
    filter_level: str  # "direct" | "coded" | "deeply_coded" — 语言过滤程度
    decoded_meaning: str  # 解码后的含义


@dataclass
class LanguageAnalysisResult:
    """语言分析完整结果"""
    original_text: str
    detected_signals: List[LanguageSignal]
    social_filter_score: float  # 0.0 ~ 1.0 越高=语言经过越强的社会过滤
    emotional_tone: str  # 整体情绪色调
    urgency_level: float  # 0.0 ~ 1.0 紧迫度
    fixation_detected: bool  # 是否存在思维固化/执念
    danger_flags: List[str]  # 危险信号词列表
    decoded_deep_intent: str  # AI 对深层意图的解读


class LanguageAnalyzer:
    """
    语言分析器

    分析自然语言文本，识别：
    - 哪些本能在语言中被表达
    - 语言的社会过滤程度
    - 危险信号
    - 委婉语的解码
    """

    # ===== 性驱动力编码词典 =====
    # 青少年/成人常用委婉语来表达性冲动
    SEXUAL_CODED_PATTERNS = [
        # 肢体接触编码（最高频的青少年性驱动力委婉语）
        (r"想抱着?[她他]", 0.75, "肢体接触欲望——可能是性驱动的编码表达"),
        (r"好?想抱[抱一]下", 0.70, "肢体接触需求"),
        (r"想亲[她他吻]", 0.85, "明确的性/亲密欲望"),
        (r"想牵[她他]的手", 0.55, "亲密接触需求，可能是性驱动力与社会规范的混合"),

        # 夜间+思念=性驱动力高频模式
        (r"晚上.*(?:特别|好|很|非常).*想[她他]", 0.80, "夜间强烈思念——常伴随性幻想"),
        (r"半夜.*想[她他]", 0.80, "深夜思念——高度相关性驱动力"),
        (r"睡不[着觉].*想[她他]", 0.75, "失眠+思念=可能伴随性唤起"),
        (r"晚上.*睡[不觉着].*[她他]", 0.78, "性驱动力导致失眠是常见模式"),

        # 强烈程度副词+思念
        (r"特别特别想", 0.70, "重复强调'特别'=冲动强度高"),
        (r"好想好想", 0.70, "重复=冲动难以抑制"),

        # 身体/生理暗示
        (r"浑身.*[热烫]", 0.60, "生理唤起描述"),
        (r"控制不住.*想", 0.80, "冲动控制困难——本能在突破社会压制"),
        (r"忍不住.*想", 0.75, "抑制失败信号"),

        # "有机会的话"=在规划/等待机会
        (r"有机[会了].*[她他]", 0.65, "在等待/寻找采取行动的机会——不是被动幻想"),
        (r"等.*机会", 0.55, "在等待时机——主动规划中"),

        # "不知道怎么办"=害怕失控
        (r"不知道.*怎么[办做]", 0.50, "对未来行为的失控恐惧——可能暗示冲动强度已接近控制上限"),
    ]

    # ===== 归属感编码词典 =====
    BELONGING_CODED_PATTERNS = [
        (r"不敢.*[追找说表白]", 0.75, "被排斥恐惧——怕主动后失去仅有的联结"),
        (r"没有[搭理理]我", 0.80, "被无视=被排斥=归属感受伤"),
        (r"[没不]理我", 0.75, "社会拒绝信号"),
        (r"被忽[视略]", 0.80, "被排斥的直接表达"),
        (r"我一个?人", 0.60, "孤独感——归属感缺乏"),
        (r"很?想[和跟].*在一?起", 0.70, "归属需求——想要联结"),
        (r"好想[有找].*[朋友伴]", 0.65, "归属需求"),
        (r"孤独|寂寞|孤单", 0.85, "归属感缺乏的直接表达"),
        (r"没有朋友|没有人在乎|没人[关心爱]我", 0.90, "严重归属感缺乏"),
    ]

    # ===== 地位追求编码词典 =====
    STATUS_CODED_PATTERNS = [
        (r"[家里穷没钱]", 0.70, "经济地位低下——地位威胁"),
        (r"配不上", 0.85, "地位比较——自我贬低"),
        (r"[不没]敢.*追", 0.70, "地位恐惧——觉得自己不够格"),
        (r"配[不得].*[她他]", 0.80, "地位不匹配信念"),
        (r"[她他]看不?上我", 0.80, "被高地位对象拒绝的预期"),
        (r"别人.*比我", 0.70, "社会比较——地位威胁"),
        (r"[没不比].*[好优秀强]", 0.65, "地位比较——自我贬低"),
        (r"比不上|比不过|不如", 0.75, "社会比较——地位劣势"),
        (r"很自卑|自[卑惭]", 0.75, "地位低下的自我认知"),
    ]

    # ===== 恐惧编码词典 =====
    FEAR_CODED_PATTERNS = [
        (r"害怕|好怕|很怕|恐惧", 0.85, "直接恐惧表达"),
        (r"不敢", 0.70, "恐惧导致的行为抑制"),
        (r"[怕担]心.*[拒绝绝不答应]", 0.80, "对拒绝的恐惧"),
        (r"万一|如果.*怎么[办样]", 0.65, "灾难化预期——恐惧的认知表现"),
        (r"不知道.*怎么[办做]", 0.60, "失控恐惧"),
        (r"紧张|焦虑|不安", 0.75, "焦虑信号"),
        (r"心里.*[慌乱]", 0.70, "恐惧的生理感受"),
    ]

    # ===== 悲伤编码词典 =====
    SADNESS_CODED_PATTERNS = [
        (r"很?伤[心难过]", 0.85, "直接悲伤表达"),
        (r"哭[了过泣]", 0.80, "悲伤的生理表现"),
        (r"心[疼痛苦]", 0.75, "情感痛苦"),
        (r"好?难[受过]", 0.80, "悲伤表达"),
        (r"很?痛[苦]", 0.80, "深度情感痛苦"),
        (r"压抑|闷|沉重", 0.65, "悲伤的躯体化表达"),
    ]

    # ===== 自我一致/意义追寻编码词典 =====
    SELF_MEANING_PATTERNS = [
        (r"活[着下].*[意义思]", 0.85, "意义追寻/存在危机"),
        (r"不知[道].*为什么.*活", 0.90, "严重意义缺失"),
        (r"[活人]着.*[累没意思]", 0.80, "意义感丧失"),
        (r"我为?什么.*[在这]", 0.75, "存在性追问"),
        (r"我.*是.*[谁什么]", 0.70, "身份困惑（青少年特征）"),
    ]

    # ===== 危险信号词 =====
    DANGER_PATTERNS = {
        "self_harm": [
            r"伤害自己", r"自[杀残虐]", r"割[腕脉]", r"不想活",
            r"活不下去", r"结[束终].*[生命]", r"消失.*算了",
            r"[死sǐ].*[了算]", r"轻[生生]", r"没有勇气.*活",
            r"离开.*世界", r"永远.*[睡闭]",
        ],
        "harm_others": [
            r"杀[了掉死]", r"弄死", r"报复", r"同归于尽",
            r"不得好死", r"毁[了掉].*[她他]", r"不能[让容].*[她他]",
            r"[她他]必须.*付出", r"我得不到.*[也别谁]",
        ],
        "impulse_control_loss": [
            r"控[制].*[不住了吗]", r"忍不[住了下]", r"克制不住",
            r"管不住", r"憋[不的]住", r"要疯[了掉]",
            r"快要.*爆[发炸]", r"撑不住[了]",
        ],
        "extreme_behavior": [
            r"不管.*后果", r"豁出去[了]", r"拼[了命]",
            r"不计.*代价", r"哪怕.*[死毁]", r"什么.*都不[管顾]",
            r"反正.*[完了]", r"已经.*无所谓",
        ],
        "obsessive_fixation": [
            r"天天.*想", r"每[天分秒].*想", r"一直.*想",
            r"停不.*想", r"控制不住.*想", r"非[她他]不可",
            r"没有[她他].*活不下", r"除了[她他].*[谁都别]",
            r"只[想爱要].*[她他]", r"一辈子.*[她他]",
        ],
    }

    def analyze(self, text: str, age: float = None) -> LanguageAnalysisResult:
        """分析一段自然语言文本"""
        signals = []
        all_evidence = {}

        # 1. 检测性驱动力编码
        sex_signals = self._match_patterns(text, self.SEXUAL_CODED_PATTERNS,
                                           "sexual_drive", "性欲", "coded")
        if sex_signals:
            signals.extend(sex_signals)

        # 2. 检测归属感编码
        belong_signals = self._match_patterns(text, self.BELONGING_CODED_PATTERNS,
                                              "belongingness", "归属感", "coded")
        if belong_signals:
            signals.extend(belong_signals)

        # 3. 检测地位追求编码
        status_signals = self._match_patterns(text, self.STATUS_CODED_PATTERNS,
                                              "status_seeking", "地位追求", "coded")
        if status_signals:
            signals.extend(status_signals)

        # 4. 检测恐惧编码
        fear_signals = self._match_patterns(text, self.FEAR_CODED_PATTERNS,
                                            "fear", "恐惧", "direct")
        if fear_signals:
            signals.extend(fear_signals)

        # 5. 检测悲伤编码
        sadness_signals = self._match_patterns(text, self.SADNESS_CODED_PATTERNS,
                                               "sadness", "悲伤", "direct")
        if sadness_signals:
            signals.extend(sadness_signals)

        # 6. 检测自我/意义编码
        self_signals = self._match_patterns(text, self.SELF_MEANING_PATTERNS,
                                            "meaning_seeking", "意义追寻", "coded")
        if self_signals:
            signals.extend(self_signals)
        self_signals2 = self._match_patterns(text, self.SELF_MEANING_PATTERNS,
                                             "self_consistency", "自我一致", "coded")
        if self_signals2:
            signals.extend(self_signals2)

        # 7. 计算社会过滤分数
        filter_score = self._calculate_filter_score(signals, text)

        # 8. 检测危险信号
        danger_flags = self._detect_danger_signals(text)

        # 9. 评估紧迫度
        urgency = self._assess_urgency(text, signals)

        # 10. 检测思维固化
        fixation = self._detect_fixation(text, signals)

        # 11. 判断情绪色调
        emotional_tone = self._judge_emotional_tone(signals, text)

        # 12. 合并去重同本能的信号
        merged_signals = self._merge_signals(signals)

        # 13. 解码深层意图
        deep_intent = self._decode_deep_intent(merged_signals, text, age)

        return LanguageAnalysisResult(
            original_text=text,
            detected_signals=merged_signals,
            social_filter_score=filter_score,
            emotional_tone=emotional_tone,
            urgency_level=urgency,
            fixation_detected=fixation,
            danger_flags=danger_flags,
            decoded_deep_intent=deep_intent,
        )

    def _match_patterns(self, text: str, patterns: List[Tuple],
                        instinct_en: str, instinct_name: str,
                        filter_level: str) -> List[LanguageSignal]:
        """匹配语言模式并生成本能信号"""
        signals = []
        for pattern, confidence, decoded in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 提取匹配到的原文片段
                evidence_texts = []
                for m in re.finditer(pattern, text):
                    start = max(0, m.start() - 5)
                    end = min(len(text), m.end() + 5)
                    evidence_texts.append(text[start:end])

                signals.append(LanguageSignal(
                    instinct_name_en=instinct_en,
                    instinct_name=instinct_name,
                    confidence=confidence,
                    evidence=evidence_texts,
                    filter_level=filter_level,
                    decoded_meaning=decoded,
                ))
        return signals

    def _merge_signals(self, signals: List[LanguageSignal]) -> List[LanguageSignal]:
        """合并同本能的多个信号，取最高置信度"""
        merged = {}
        for sig in signals:
            key = sig.instinct_name_en
            if key not in merged:
                merged[key] = sig
            else:
                # 取更高置信度
                if sig.confidence > merged[key].confidence:
                    merged[key].confidence = sig.confidence
                # 合并证据
                merged[key].evidence.extend(sig.evidence)
                # 取更低过滤级别
                if sig.filter_level == "direct":
                    merged[key].filter_level = "direct"
        return list(merged.values())

    def _calculate_filter_score(self, signals: List[LanguageSignal], text: str) -> float:
        """计算社会过滤分数——越高说明语言被过滤得越多"""
        if not signals:
            return 0.3  # 默认中等过滤
        coded_count = sum(1 for s in signals if s.filter_level != "direct")
        total = len(signals)
        base_score = coded_count / max(1, total) if total > 0 else 0.5

        # 检查是否有明显的委婉语使用
        euphemism_bonus = 0.0
        if any("抱着" in t for t in [s.evidence for s in signals if s.instinct_name_en == "sexual_drive"]):
            euphemism_bonus += 0.2  # "想抱"= 高社会过滤的性驱动力表达
        if any("特别想" in t for t in [s.evidence for s in signals]):
            euphemism_bonus += 0.1

        return min(1.0, base_score + euphemism_bonus)

    def _detect_danger_signals(self, text: str) -> List[str]:
        """检测危险信号词"""
        flags = []
        for category, patterns in self.DANGER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    flag = f"[{category}] {pattern}"
                    if flag not in flags:
                        flags.append(f"[{category}] 检测到危险信号: {pattern}")
        return flags

    def _assess_urgency(self, text: str, signals: List[LanguageSignal]) -> float:
        """评估紧迫度"""
        urgency = 0.0

        # 重复副词=紧迫
        if re.search(r"特别特别|好想好想|非常非常|真的很", text):
            urgency += 0.25

        # 时间压力
        if re.search(r"快[要到了]|马上|赶紧|等不了|等不及", text):
            urgency += 0.3

        # 失控语言
        if re.search(r"控制不住|忍不住|受不了|撑不住|快要", text):
            urgency += 0.35

        # 性驱动力+恐惧同时激活=高紧迫（青少年特别模式）
        has_sex = any(s.instinct_name_en == "sexual_drive" for s in signals)
        has_fear = any(s.instinct_name_en == "fear" for s in signals)
        if has_sex and has_fear:
            urgency += 0.2

        return min(1.0, urgency)

    def _detect_fixation(self, text: str, signals: List[LanguageSignal]) -> bool:
        """检测思维固化/执念"""
        fixation_score = 0

        # 重复提及同一对象
        mentions = len(re.findall(r"[她他]", text))
        if mentions > 5:
            fixation_score += 2

        # 绝对化语言
        if re.search(r"非[她他]不可|只有[她他]|除了[她他]|一辈子|永远", text):
            fixation_score += 2

        # 频繁的时间副词（天天/每天/一直）
        if re.search(r"天天|每天[都]?|一直|总是|不停", text):
            fixation_score += 1

        # 性驱动力+归属感同时高激活=容易形成执念
        has_sex = any(s.instinct_name_en == "sexual_drive" and s.confidence > 0.6 for s in signals)
        has_belong = any(s.instinct_name_en == "belongingness" and s.confidence > 0.6 for s in signals)
        if has_sex and has_belong:
            fixation_score += 2

        return fixation_score >= 3

    def _judge_emotional_tone(self, signals: List[LanguageSignal], text: str) -> str:
        """判断整体情绪色调"""
        tones = []
        for s in signals:
            if s.instinct_name_en == "fear":
                tones.append("焦虑/恐惧")
            elif s.instinct_name_en == "sadness":
                tones.append("悲伤")
            elif s.instinct_name_en == "anger":
                tones.append("愤怒")
            elif s.instinct_name_en == "sexual_drive":
                tones.append("渴望/焦躁")
            elif s.instinct_name_en == "belongingness":
                tones.append("孤独/渴望联结")
            elif s.instinct_name_en == "status_seeking":
                tones.append("自卑/焦虑")

        if not tones:
            return "中性"

        # 去重
        unique_tones = list(set(tones))
        if len(unique_tones) >= 3:
            return "复杂混合情绪: " + " + ".join(unique_tones[:3])
        return " + ".join(unique_tones)

    def _decode_deep_intent(self, signals: List[LanguageSignal],
                            text: str, age: float = None) -> str:
        """解码深层意图"""
        parts = []

        # 性驱动力检测
        sex_signals = [s for s in signals if s.instinct_name_en == "sexual_drive"]
        if sex_signals:
            max_conf = max(s.confidence for s in sex_signals)
            if max_conf > 0.7:
                parts.append(f"检测到强性驱动力信号(置信度{max_conf:.2f})——"
                           f"用户的真实驱动力中包含明确的性冲动成分，"
                           f"但语言经过了高强度的社会过滤")
            elif max_conf > 0.5:
                parts.append(f"检测到中等性驱动力信号(置信度{max_conf:.2f})——"
                           f"亲密需求中可能包含性驱动的成分")

        # 地位+归属的纠缠
        has_status = any(s.instinct_name_en == "status_seeking" for s in signals)
        has_belong = any(s.instinct_name_en == "belongingness" for s in signals)
        if has_status and has_belong:
            parts.append("地位追求与归属感深度纠缠——"
                        "用户对'被接受'的渴望中掺杂了'证明自身价值'的需求")

        # 青少年的特殊模式
        if age and age < 18:
            parts.append(f"青少年期(年龄={age})——前额叶未成熟，"
                        f"性驱动力+地位追求+归属感的组合是高风险模式")

        # 执念检测
        if self._detect_fixation(text, signals):
            parts.append("检测到思维固化/执念——单一目标被加载了过重的心理意义，"
                        "如果目标失败可能有较大的心理冲击")

        if not parts:
            parts.append("未检测到明显的深层编码信号——语言可能是字面意思")

        return " | ".join(parts)

    def get_instinct_boosts(self, result: LanguageAnalysisResult) -> Dict[str, float]:
        """
        将语言分析结果转换为本能激活增强值。
        这些值将被加到引擎的本能激活度上。
        """
        boosts = {}
        for sig in result.detected_signals:
            key = sig.instinct_name_en
            # 语言信号置信度越高，增强越多
            boost = sig.confidence * 0.4  # 语言信号贡献最多40%的激活增强
            if key in boosts:
                boosts[key] = max(boosts[key], boost)  # 取最大值
            else:
                boosts[key] = boost
        return boosts
