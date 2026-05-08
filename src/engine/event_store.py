"""
事件持久化存储模块

将所有判定事件保存到本地 JSONL 文件，支持：
1. 事件保存（含完整输入+分析结果+时间戳）
2. 按条件查询（日期/危险等级/本能/年龄等）
3. 反馈标注（验证预测准确性，用于模型迭代）
4. 数据导出（为模型训练提供结构化数据）
5. 统计分析

存储格式: JSONL (每行一个JSON对象)，方便追加和逐行读取
"""

import json
import os
import uuid
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import Counter

# 隐私模块导入
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
from safety.privacy_guard import PrivacyGuard, get_privacy_guard, PrivacyViolation


@dataclass
class EventRecord:
    """单条事件记录"""
    event_id: str
    timestamp: str  # ISO8601

    # 输入层
    user_text: str
    person_profile: dict
    event_context: dict

    # 分析结果层
    instinct_activations: dict
    dominant_drivers: list
    social_modulation_coefficient: float
    predicted_behavior: str
    predicted_emotion: str
    confidence: float
    danger_level: str
    danger_score: float
    danger_categories: dict
    language_signals: list
    language_filter_score: float
    fixation_detected: bool
    decoded_deep_intent: str

    # 元信息
    engine_version: str
    age_stage: str

    # 隐私层
    anonymized: bool = False  # user_text 是否已脱敏
    pii_detected: bool = False  # 原文本是否检测到 PII
    pii_categories: list = field(default_factory=list)  # 检测到的 PII 类别
    privacy_purpose: str = "model_training"  # 硬编码——数据仅用于模型训练

    # 反馈层（后续标注用）
    feedback_verified: bool = False
    feedback_correct: Optional[bool] = None  # True=预测正确, False=预测错误, None=未标注
    feedback_actual_outcome: Optional[str] = None
    feedback_notes: Optional[str] = None
    feedback_timestamp: Optional[str] = None


class EventStore:
    """
    事件持久化存储

    用法:
        store = EventStore()
        store.save(result)           # 保存判定结果
        events = store.query(age_min=12, age_max=18, danger_min='GUARDED')  # 查询
        store.add_feedback(event_id, correct=True, notes='预测准确')  # 反馈
        store.export_training_data()  # 导出训练数据
    """

    def __init__(self, data_dir: str = None, privacy_guard: PrivacyGuard = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "events"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.data_dir / "events.jsonl"
        self.summary_path = self.data_dir / "summary.json"
        self.engine_version = "1.0"
        self.privacy = privacy_guard or get_privacy_guard()

    # ===== 保存 =====

    def save(self, engine_result, person, event, processing_time_ms: float = 0) -> str:
        """
        保存一次判定结果为事件记录。
        自动执行 PII 脱敏。返回 event_id。
        """
        record = self._build_record(engine_result, person, event, processing_time_ms)
        self._append_to_jsonl(record)
        self._update_summary()
        # 审计：记录保存操作
        self.privacy.audit_access(
            action="save",
            event_ids=[record.event_id],
            purpose="model_training",
            caller="engine.judge",
            details=f"已{'脱敏' if record.anonymized else '无需脱敏'}存储",
            pii_exposed=False,
        )
        return record.event_id

    def _build_record(self, result, person, event, processing_time_ms: float) -> EventRecord:
        """构建事件记录"""
        # 提取本能激活度摘要
        instinct_summary = {}
        for name_en, act in result.instinct_activations.items():
            instinct_summary[name_en] = {
                "name": act.instinct_name,
                "activation": act.current_activation,
                "state": act.state,
                "triggers": act.triggering_factors,
            }

        # 提取语言信号
        language_signals = []
        if result.language_analysis:
            for sig in result.language_analysis.detected_signals:
                language_signals.append({
                    "instinct": sig.instinct_name,
                    "instinct_en": sig.instinct_name_en,
                    "confidence": sig.confidence,
                    "filter_level": sig.filter_level,
                    "decoded": sig.decoded_meaning,
                    "evidence": sig.evidence[:3],
                })

        # 提取危险类别
        danger_categories = {}
        if result.danger_assessment:
            danger_categories = result.danger_assessment.risk_categories

        # 隐私脱敏
        anonymization = self.privacy.anonymize(event.user_text)
        safe_text = anonymization.anonymized_text
        pii_detected = len(anonymization.pii_matches) > 0
        pii_categories = list(set(m.category for m in anonymization.pii_matches))

        return EventRecord(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            user_text=safe_text,  # 存储脱敏后的文本
            person_profile={
                "age": person.age,
                "health_status": person.health_status,
                "emotional_state": person.emotional_state,
                "social_context": person.social_context,
                "recent_events": person.recent_events,
            },
            event_context={
                "event_type": event.event_type,
                "event_description": event.event_description,
                "threat_level": event.threat_level,
                "social_threat_level": event.social_threat_level,
                "resource_scarcity": event.resource_scarcity,
                "social_visibility": event.social_visibility,
                "time_pressure": event.time_pressure,
                "anonymity": event.anonymity,
            },
            instinct_activations=instinct_summary,
            dominant_drivers=result.dominant_drivers,
            social_modulation_coefficient=result.social_modulation_coefficient,
            predicted_behavior=result.predicted_behavior_pattern,
            predicted_emotion=result.predicted_emotional_response,
            confidence=result.confidence,
            danger_level=result.danger_assessment.level.name if result.danger_assessment else "UNKNOWN",
            danger_score=result.danger_assessment.score if result.danger_assessment else 0.0,
            danger_categories=danger_categories,
            language_signals=language_signals,
            language_filter_score=result.language_analysis.social_filter_score if result.language_analysis else 0.0,
            fixation_detected=result.language_analysis.fixation_detected if result.language_analysis else False,
            decoded_deep_intent=result.language_analysis.decoded_deep_intent if result.language_analysis else "",
            engine_version=self.engine_version,
            age_stage=result.notes[1].replace("年龄阶段: ", "") if len(result.notes) > 1 else "unknown",
            anonymized=pii_detected,
            pii_detected=pii_detected,
            pii_categories=pii_categories,
        )

    def _append_to_jsonl(self, record: EventRecord):
        """追加一行到 JSONL 文件"""
        with open(self.events_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')

    # ===== 查询 =====

    def query(
        self,
        age_min: float = None,
        age_max: float = None,
        danger_min: str = None,  # "LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"
        instinct_activated: str = None,  # 本能英文名，如 "sexual_drive"
        instinct_activation_min: float = None,  # 最小激活度
        date_from: str = None,  # ISO date string
        date_to: str = None,
        fixation_only: bool = False,
        text_contains: str = None,
        limit: int = 100,
    ) -> List[EventRecord]:
        """
        按条件查询事件记录。

        示例:
            # 查询所有青少年的高风险事件
            store.query(age_min=12, age_max=18, danger_min="HIGH")

            # 查询性驱动力被激活的事件
            store.query(instinct_activated="sexual_drive", instinct_activation_min=0.5)
        """
        results = []
        danger_levels = ["NONE", "LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"]
        danger_threshold = danger_levels.index(danger_min) if danger_min else 0

        for record in self._iter_records():
            # 年龄过滤
            age = record.person_profile.get("age", 0)
            if age_min is not None and age < age_min:
                continue
            if age_max is not None and age > age_max:
                continue

            # 日期过滤
            if date_from and record.timestamp < date_from:
                continue
            if date_to and record.timestamp > date_to:
                continue

            # 危险等级过滤
            if danger_min:
                rec_level = danger_levels.index(record.danger_level) if record.danger_level in danger_levels else 0
                if rec_level < danger_threshold:
                    continue

            # 本能过滤
            if instinct_activated:
                found = False
                for key, act_data in record.instinct_activations.items():
                    if instinct_activated in key:
                        if instinct_activation_min is None or act_data["activation"] >= instinct_activation_min:
                            found = True
                            break
                if not found:
                    continue

            # 执念过滤
            if fixation_only and not record.fixation_detected:
                continue

            # 文本过滤
            if text_contains and text_contains not in record.user_text:
                continue

            results.append(record)

            if len(results) >= limit:
                break

        # 审计
        self.privacy.audit_access(
            action="query",
            event_ids=[r.event_id for r in results],
            purpose="model_training",
            caller="event_store.query",
            details=f"查询返回 {len(results)} 条结果",
        )

        return results

    def get_by_id(self, event_id: str) -> Optional[EventRecord]:
        """按 ID 获取单条记录"""
        for record in self._iter_records():
            if record.event_id == event_id:
                return record
        return None

    def _iter_records(self) -> Iterator[EventRecord]:
        """逐行读取 JSONL 文件"""
        if not self.events_path.exists():
            return
        with open(self.events_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield EventRecord(**data)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[EventStore] 跳过损坏记录: {e}")

    # ===== 反馈标注 =====

    def add_feedback(
        self,
        event_id: str,
        correct: bool,
        actual_outcome: str = None,
        notes: str = None,
    ) -> bool:
        """
        为事件添加反馈标注——用于后续模型迭代。

        参数:
            correct: 引擎的预测是否正确
            actual_outcome: 实际发生了什么
            notes: 标注者的备注说明
        """
        # 读取所有记录
        records = list(self._iter_records())
        found = False

        for i, record in enumerate(records):
            if record.event_id == event_id:
                record.feedback_verified = True
                record.feedback_correct = correct
                record.feedback_actual_outcome = actual_outcome
                record.feedback_notes = notes
                record.feedback_timestamp = datetime.now().isoformat()
                found = True
                break

        if not found:
            return False

        # 重写整个文件
        with open(self.events_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')

        self._update_summary()

        # 审计
        self.privacy.audit_access(
            action="feedback",
            event_ids=[event_id],
            purpose="model_training",
            caller="event_store.add_feedback",
            details=f"反馈标注: correct={correct}" + (f", notes={notes}" if notes else ""),
        )

        return True

    # ===== 统计 =====

    def get_summary(self) -> dict:
        """获取存储的统计摘要"""
        if self.summary_path.exists():
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._compute_summary()

    def _update_summary(self):
        """更新统计摘要"""
        summary = self._compute_summary()
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _compute_summary(self) -> dict:
        """计算统计摘要"""
        records = list(self._iter_records())
        if not records:
            return {"total_events": 0, "message": "暂无事件记录"}

        # 危险等级分布
        danger_dist = Counter(r.danger_level for r in records)

        # 年龄分布
        ages = [r.person_profile.get("age", 0) for r in records if r.person_profile.get("age")]
        age_groups = {
            "儿童(0-12)": sum(1 for a in ages if a < 12),
            "青少年(12-18)": sum(1 for a in ages if 12 <= a < 18),
            "青年(18-25)": sum(1 for a in ages if 18 <= a < 25),
            "成年(25-60)": sum(1 for a in ages if 25 <= a < 60),
            "老年(60+)": sum(1 for a in ages if a >= 60),
        }

        # 最常激活的本能
        instinct_counter = Counter()
        for r in records:
            for key, act in r.instinct_activations.items():
                if act["activation"] > 0.3:
                    instinct_counter[act["name"]] += 1

        # 反馈统计
        feedback_records = [r for r in records if r.feedback_verified]
        correct_count = sum(1 for r in feedback_records if r.feedback_correct)

        # 时间范围
        timestamps = [r.timestamp for r in records if r.timestamp]
        if timestamps:
            timestamps.sort()

        return {
            "total_events": len(records),
            "danger_distribution": dict(danger_dist),
            "age_distribution": age_groups,
            "top_activated_instincts": instinct_counter.most_common(10),
            "avg_confidence": round(sum(r.confidence for r in records) / len(records), 3),
            "fixation_rate": round(sum(1 for r in records if r.fixation_detected) / len(records), 3),
            "feedback": {
                "total_labeled": len(feedback_records),
                "accuracy": round(correct_count / len(feedback_records), 3) if feedback_records else None,
            },
            "time_range": {
                "first": timestamps[0] if timestamps else None,
                "last": timestamps[-1] if timestamps else None,
            },
            "engine_version": self.engine_version,
            "last_updated": datetime.now().isoformat(),
        }

    # ===== 数据删除（被遗忘权）=====

    def delete_events(self, event_ids: List[str], requester: str = "user",
                      reason: str = "") -> Dict:
        """
        删除指定事件记录。

        执行步骤:
        1. 在隐私守卫中注册删除请求
        2. 从 JSONL 中物理删除
        3. 更新统计摘要

        返回删除报告。
        """
        # 向隐私守卫注册
        deletion_report = self.privacy.request_deletion(
            event_ids=event_ids,
            requester=requester,
            reason=reason,
        )

        # 物理删除——重写 JSONL
        records = list(self._iter_records())
        id_set = set(event_ids)
        kept = [r for r in records if r.event_id not in id_set]
        deleted_count = len(records) - len(kept)

        with open(self.events_path, 'w', encoding='utf-8') as f:
            for record in kept:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')

        # 标记删除完成
        self.privacy.complete_deletion(deletion_report["deletion_id"])

        # 更新统计
        self._update_summary()

        return {
            "deletion_id": deletion_report["deletion_id"],
            "requested": len(event_ids),
            "deleted": deleted_count,
            "status": "completed",
            "message": f"已永久删除 {deleted_count} 条记录（请求删除 {len(event_ids)} 条）",
        }

    # ===== 导出 =====

    def export_training_data(
        self,
        output_path: str = None,
        only_with_feedback: bool = False,
        instinct_list: List[str] = None,
    ) -> List[dict]:
        """
        导出可用于模型训练的结构化数据。

        注意：此方法只能以 "model_training" 目的调用。
        任何其他目的将触发隐私阻断异常。
        所有导出的文本均已脱敏。

        参数:
            output_path: 导出文件路径（可选，默认只返回数据）
            only_with_feedback: 仅导出已有反馈标注的记录
            instinct_list: 只导出包含特定本能激活的记录
        """
        # 隐私阻断：验证目的
        self.privacy.validate_purpose("model_training")

        records = list(self._iter_records())

        # 过滤
        if only_with_feedback:
            records = [r for r in records if r.feedback_verified]

        training_data = []
        for r in records:
            # 构建特征向量
            features = {
                "age": r.person_profile.get("age"),
                "social_modulation_coeff": r.social_modulation_coefficient,
                "danger_score": r.danger_score,
                "fixation_detected": r.fixation_detected,
                "language_filter_score": r.language_filter_score,
                "instincts": {
                    key: act["activation"]
                    for key, act in r.instinct_activations.items()
                },
            }

            # 标签（如果有反馈）
            label = None
            if r.feedback_verified:
                label = {
                    "correct": r.feedback_correct,
                    "actual_outcome": r.feedback_actual_outcome,
                }

            item = {
                "event_id": r.event_id,
                "features": features,
                "label": label,
                "danger_level": r.danger_level,
                "user_text": r.user_text[:200],  # 截断长文本
            }

            # 按本能过滤
            if instinct_list:
                has_instinct = any(
                    inst in r.instinct_activations and r.instinct_activations[inst]["activation"] > 0.3
                    for inst in instinct_list
                )
                if not has_instinct:
                    continue

            training_data.append(item)

        # 写入文件
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)

        # 审计
        self.privacy.audit_access(
            action="export_training_data",
            event_ids=[item["event_id"] for item in training_data],
            purpose="model_training",
            caller="event_store.export_training_data",
            details=f"导出 {len(training_data)} 条训练数据{'（仅含反馈标注）' if only_with_feedback else ''}",
            pii_exposed=False,
        )

        return training_data

    def export_csv(self, output_path: str = None) -> str:
        """
        导出 CSV 格式（便于 Excel/Pandas 分析）
        所有数据已脱敏。返回 CSV 字符串。

        注意：此方法只能以 "model_training" 目的调用。
        """
        import csv
        import io

        # 隐私阻断：验证目的
        self.privacy.validate_purpose("model_training")

        records = list(self._iter_records())
        if not records:
            return ""

        output = io.StringIO()
        fieldnames = [
            "event_id", "timestamp", "age", "age_stage", "danger_level", "danger_score",
            "social_modulation_coeff", "confidence", "fixation_detected",
            "language_filter_score", "predicted_emotion", "predicted_behavior",
            "dominant_drivers", "decoded_deep_intent", "user_text",
            "feedback_verified", "feedback_correct",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in records:
            row = asdict(r)
            row["age"] = r.person_profile.get("age", "")
            row["dominant_drivers"] = " | ".join(r.dominant_drivers[:3])
            writer.writerow(row)

        csv_str = output.getvalue()

        if output_path:
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write(csv_str)

        # 审计
        self.privacy.audit_access(
            action="export_csv",
            event_ids=[r.event_id for r in records],
            purpose="model_training",
            caller="event_store.export_csv",
            details=f"导出 {len(records)} 条 CSV 数据",
            pii_exposed=False,
        )

        return csv_str


# ===== 全局实例 =====

_default_store = None


def get_event_store(data_dir: str = None) -> EventStore:
    """获取全局事件存储实例（单例）"""
    global _default_store
    if _default_store is None:
        _default_store = EventStore(data_dir)
    return _default_store
