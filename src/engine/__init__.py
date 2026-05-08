from .human_thinking_engine import (
    HumanThinkingEngine,
    PersonProfile,
    EventContext,
    JudgmentResult,
    InstinctActivation,
    create_person,
    create_event,
)
from .event_store import EventStore, get_event_store
from .language_analyzer import LanguageAnalyzer, LanguageAnalysisResult
from .danger_assessor import DangerAssessor, DangerAssessment, DangerLevel
from .cognitive_engine import (
    CognitiveEngine,
    CognitiveAnalysisResult,
    WorldviewInference,
    LanguageStyleAnalysisResult,
    CognitiveBiasResult,
    DefenseMechanismResult,
    IntentHypothesis,
)

# 隐私模块（从 safety 包导入）
import sys
from pathlib import Path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
from safety.privacy_guard import PrivacyGuard, get_privacy_guard, PrivacyViolation
