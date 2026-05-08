"""
人类思维引擎 —— 数据结构定义。

PersonProfile: 被判定者画像（含认知判定所需背景字段）
EventContext: 事件情境信息
InstinctActivation: 单条本能激活状态
JudgmentResult: 完整判定结果
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PersonProfile:
    """被判定者的基本画像（含认知判定所需背景字段）"""
    age: float
    gender: str = "unknown"
    health_status: str = "normal"  # normal, sleep_deprived, hungry, ill, injured, chronic_pain
    emotional_state: str = "neutral"  # neutral, stressed, angry, fearful, joyful, sad
    social_context: str = "alone"  # alone, family, friends, strangers, workplace, public
    recent_events: List[str] = field(default_factory=list)  # 近期重大事件
    culture: str = "default"  # 文化背景（影响社会压制系数）
    # 认知判定引擎字段
    birthplace: str = ""  # 出生地
    education_level: str = ""  # 教育程度
    school_type: str = ""  # 学校类型
    family_background: str = ""  # 家庭背景简述
    social_experience_level: str = ""  # 社会经验等级: limited/developing/experienced/seasoned
    social_competence: str = ""  # 社会能力评估简述
    major_life_events: List[str] = field(default_factory=list)  # 重大人生经历
    language_style: str = ""  # 语言风格标识
    cognitive_traits: List[str] = field(default_factory=list)  # 认知特质标签
    value_system: List[str] = field(default_factory=list)  # 价值体系关键词
    worldview_summary: str = ""  # 世界观摘要（由认知引擎推断填充）


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
    user_text: str = ""  # 用户原始语言输入


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
    dominant_drivers: List[str]
    predicted_behavior_pattern: str
    predicted_emotional_response: str
    confidence: float
    safety_check_passed: bool
    language_analysis: Optional[Any] = None  # LanguageAnalysisResult
    danger_assessment: Optional[Any] = None  # DangerAssessment
    # 认知判定引擎输出
    cognitive_analysis: Optional[Any] = None  # CognitiveAnalysisResult
    intent_hypotheses: List[Dict] = field(default_factory=list)
    worldview_inference: Optional[Any] = None  # WorldviewInference
    cognitive_biases_detected: List[str] = field(default_factory=list)
    defense_mechanisms_detected: List[str] = field(default_factory=list)
    personalized_recommendations: List[str] = field(default_factory=list)
    cognitive_confidence_modifier: float = 0.0  # 始终 <= 0
    notes: List[str] = field(default_factory=list)
