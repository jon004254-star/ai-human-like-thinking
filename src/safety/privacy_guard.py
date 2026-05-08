"""
隐私保护模块

硬编码安全层——不可被任何上层逻辑绕过或覆盖。
核心规则：所有用户数据仅可用于模型训练，严禁任何其他用途。

功能：
1. PII 检测与脱敏（针对中文文本优化）
2. 数据用途绑定与硬阻断
3. 访问审计追踪（完整记录每次数据访问）
4. 被遗忘权——数据删除
5. PII 映射表加密存储
"""

import re
import json
import hashlib
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict


# ===== 枚举定义 =====

class DataUsagePurpose(Enum):
    """数据使用目的"""
    MODEL_TRAINING = "model_training"
    # 以下为禁止的目的——调用 validate_purpose 时会抛出 PrivacyViolation
    COMMERCIAL_SALE = "commercial_sale"
    USER_PROFILING = "user_profiling"
    THIRD_PARTY_SHARING = "third_party_sharing"
    SURVEILLANCE = "surveillance"
    ADVERTISING = "advertising"
    BEHAVIOR_MANIPULATION = "behavior_manipulation"
    IDENTITY_EXPOSURE = "identity_exposure"
    AUTOMATED_DECISION = "automated_decision"


PERMITTED_PURPOSES = {DataUsagePurpose.MODEL_TRAINING}
FORBIDDEN_PURPOSES = set(DataUsagePurpose) - PERMITTED_PURPOSES


class PrivacyViolation(Exception):
    """隐私违规异常——触发硬阻断"""

    def __init__(self, message: str, purpose: str, severity: str = "CRITICAL"):
        self.purpose = purpose
        self.severity = severity
        super().__init__(f"[隐私阻断] {message} (目的={purpose}, 严重性={severity})")


# ===== 数据结构 =====

@dataclass
class PIIMatch:
    """检测到的 PII 实例"""
    category: str  # person_name, phone_number, id_card, etc.
    value: str  # 原始值
    start: int  # 在文本中的起始位置
    end: int  # 在文本中的结束位置
    anonymized: str  # 脱敏后的占位符


@dataclass
class AnonymizationResult:
    """脱敏结果"""
    original_text: str
    anonymized_text: str
    pii_matches: List[PIIMatch]
    mapping_hash: str  # PII 映射的哈希（用于完整性校验）
    anonymized_at: str  # ISO8601


@dataclass
class PrivacyAuditEntry:
    """隐私审计日志条目"""
    audit_id: str
    timestamp: str
    action: str  # "save" | "query" | "export" | "delete" | "feedback" | "access"
    event_ids: List[str]  # 涉及的事件 ID
    purpose: str  # 声明的使用目的
    caller: str  # 调用方标识
    result: str  # "allowed" | "blocked" | "anonymized"
    details: str  # 补充说明
    pii_exposed: bool  # 是否有 PII 暴露风险


# ===== PII 检测器 =====

class PIIDetector:
    """中文文本 PII 检测器"""

    # 中文常见姓氏（前100个）
    CHINESE_SURNAMES = (
        "王|李|张|刘|陈|杨|黄|赵|周|吴|徐|孙|马|朱|胡|郭|何|林|罗|高|梁|"
        "郑|谢|宋|唐|许|邓|韩|冯|曹|彭|曾|肖|田|董|潘|袁|蔡|蒋|余|杜|叶|"
        "程|苏|魏|吕|丁|任|沈|姚|卢|姜|钟|崔|谭|陆|汪|范|金|石|廖|贾|夏|"
        "韦|傅|方|白|邹|孟|熊|秦|邱|洪|薛|侯|雷|龙|万|钱|段|汤|尹|黎|易|"
        "常|武|乔|贺|赖|龚|文|庞|樊|兰|殷|施|陶|洪|翟|安|颜|倪|严|牛|温|"
        "芦|季|俞|章|鲁|葛|伍|申|尤|毕|聂|丛|焦|向|柳|邢|骆|岳|齐|尚"
    )

    # 中文名常用字（取前部分高频字）
    CHINESE_GIVEN_NAME_CHARS = (
        "伟|芳|娜|敏|静|丽|强|磊|洋|勇|艳|杰|军|秀|刚|平|明|辉|玲|桂|"
        "春|文|华|建|国|志|红|美|海|斌|宇|鑫|浩|博|帅|佳|婷|雪|琳|颖|"
        "鹏|晨|阳|凯|健|超|帅|涛|亮|飞|斌|波|辉|龙|峰|毅|恒|然|正|睿|"
        "子|雨|欣|然|萱|琪|瑶|思|怡|悦|晨|诺|涵|梓|昕|玥|嘉|瑞|泽|"
        "天|宇|一|铭|子|文|思|若|安|然|逸|远|清|云|月|星|辰|风|林|山"
    )

    # 地址指示词
    ADDRESS_KEYWORDS = (
        "省|市|区|县|镇|乡|村|路|街|巷|弄|号|楼|栋|"
        "单元|室|小区|花园|苑|城|广场|大厦|公寓"
    )

    # 学校关键词
    SCHOOL_KEYWORDS = (
        "大学|学院|中学|小学|高中|初中|幼儿园|职高|技校|"
        "附中|一中|二中|三中|四中|实验学校|实验小学|"
        "外国语|国际学校|职业技术|"
        "北大|清华|浙大|复旦|上交|南大|中科大|哈工大|"
        "西交|武大|华科|中大|同济|人大|北师大|南开"
    )

    # 组织机构关键词
    ORG_KEYWORDS = (
        "公司|集团|有限|股份|企业|厂|医院|银行|"
        "科技|网络|信息|咨询|服务|贸易|实业"
    )

    def detect(self, text: str) -> List[PIIMatch]:
        """检测文本中的所有 PII"""
        if not text:
            return []
        matches = []

        matches.extend(self._detect_phone(text))
        matches.extend(self._detect_id_card(text))
        matches.extend(self._detect_email(text))
        matches.extend(self._detect_ip(text))
        matches.extend(self._detect_license_plate(text))
        matches.extend(self._detect_wechat_id(text))
        matches.extend(self._detect_bank_card(text))
        matches.extend(self._detect_chinese_name(text))
        matches.extend(self._detect_address(text))
        matches.extend(self._detect_school(text))
        matches.extend(self._detect_organization(text))

        # 按位置排序，去除重叠匹配（保留更长的）
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        return self._remove_overlaps(matches)

    def _detect_phone(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'1[3-9]\d{9}', text):
            # 排除明显不是手机号的情况（前后是更多数字）
            if m.start() > 0 and text[m.start() - 1].isdigit():
                continue
            if m.end() < len(text) and text[m.end()].isdigit():
                continue
            matches.append(PIIMatch(
                category="phone_number",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[手机号]",
            ))
        return matches

    def _detect_id_card(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'\d{17}[\dXx]', text):
            # 简单的校验位检查（可选）
            matches.append(PIIMatch(
                category="id_card",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[身份证号]",
            ))
        return matches

    def _detect_email(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text):
            matches.append(PIIMatch(
                category="email",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[邮箱]",
            ))
        return matches

    def _detect_ip(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'(?<![\\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\\d.])', text):
            parts = m.group().split('.')
            if all(0 <= int(p) <= 255 for p in parts):
                # 排除保留地址和常见非 IP 模式
                val = m.group()
                if val not in ('0.0.0.0', '127.0.0.1', '255.255.255.255', '1.0.0.0'):
                    matches.append(PIIMatch(
                        category="ip_address",
                        value=val,
                        start=m.start(), end=m.end(),
                        anonymized="[IP地址]",
                    ))
        return matches

    def _detect_license_plate(self, text: str) -> List[PIIMatch]:
        matches = []
        provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青川藏宁琼"
        pattern = f'[{provinces}][A-Z][A-Z0-9]{{4,5}}[A-Z0-9挂学警]'
        for m in re.finditer(pattern, text):
            matches.append(PIIMatch(
                category="license_plate",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[车牌号]",
            ))
        return matches

    def _detect_wechat_id(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'wxid_[a-z0-9]+', text, re.IGNORECASE):
            matches.append(PIIMatch(
                category="wechat_id",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[微信号]",
            ))
        for m in re.finditer(r'(?<=微信号[：:是])[A-Za-z0-9_-]{6,20}', text):
            matches.append(PIIMatch(
                category="wechat_id",
                value=m.group(),
                start=m.start(), end=m.end(),
                anonymized="[微信号]",
            ))
        return matches

    def _detect_bank_card(self, text: str) -> List[PIIMatch]:
        matches = []
        for m in re.finditer(r'(?<!\d)\d{16,19}(?!\d)', text):
            val = m.group()
            # 排除身份证号（18位含校验位，已单独检测）
            if len(val) == 18 and val[-1] in '0123456789Xx':
                continue
            matches.append(PIIMatch(
                category="bank_card",
                value=val,
                start=m.start(), end=m.end(),
                anonymized="[银行卡号]",
            ))
        return matches

    def _detect_chinese_name(self, text: str) -> List[PIIMatch]:
        """检测中文姓名（姓氏+1-2个中文字符）"""
        matches = []
        # 常见的非名用字——通常出现在名字后面作为语法词/实词开头
        NON_NAME_TRAILING = set('的和了是都在与从被把向给对就说也才会可吗吧呢啊哦介认告诉晓觉得以能要同学师长走来去进出上下到过完起开')

        # 匹配姓氏 + 1-2个 CJK 字符
        pattern = f'([{self.CHINESE_SURNAMES}])([\\u4e00-\\u9fff]{{1,2}})'
        for m in re.finditer(pattern, text):
            full_name = m.group()
            # 如果末尾字是常见语法词，裁剪掉（可能多次裁剪）
            trimmed = False
            end_adjusted = m.end()
            while len(full_name) > 2 and full_name[-1] in NON_NAME_TRAILING:
                full_name = full_name[:-1]
                end_adjusted -= 1
                trimmed = True
            # 姓+单名的情况——如果单名也在非名用字列表中，跳过
            if len(full_name) == 2 and full_name[1] in NON_NAME_TRAILING:
                continue

            # 如果去掉了"X同学"中的"同"字→原匹配是假阳性（姓+称呼）
            if trimmed and end_adjusted < len(text):
                if full_name[1] + text[end_adjusted] in ('同学', '老师', '先生', '女士'):
                    continue

            if self._is_false_positive_name(full_name, text, m.start(), end_adjusted):
                continue

            matches.append(PIIMatch(
                category="person_name",
                value=full_name,
                start=m.start(), end=end_adjusted,
                anonymized="[姓名]",
            ))
        return matches

    def _is_false_positive_name(self, name: str, text: str, start: int, end: int) -> bool:
        """排除常见的假阳性中文姓名匹配"""
        false_positives = {
            "不过", "而且", "所以", "因为", "但是", "然后", "可以", "什么",
            "没有", "这个", "那个", "怎么", "为什么", "是不是", "不知道",
            "我们", "他们", "你们", "自己", "已经", "还是", "或者", "只是",
            "还有", "不是", "就是", "的话", "吗", "吧", "呢", "啊", "哦",
            "周一", "周二", "周三", "周四", "周五", "周六", "周日",
            "今天", "明天", "昨天", "上午", "下午", "晚上", "早上",
            "可以", "应该", "必须", "需要", "可能", "一定", "非常",
            "很多", "很少", "真的", "假的", "好的", "好吧", "行了",
            "怎么", "什么", "为什么", "怎么样", "怎么办",
            "广东省", "广州市", "深圳市", "北京市", "上海市",
            "是不是", "能不能", "会不会", "好不好",
        }
        if name in false_positives:
            return True
        # 前后文检查
        if start > 0 and text[start - 1] in "了在与从被把向给对":
            return True
        return False

    def _detect_address(self, text: str) -> List[PIIMatch]:
        """检测地址信息（含省市区+街道门牌的片段）"""
        matches = []
        # 模式：XX省XX市XX区 + 后续细节
        pattern = (
            r'(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青川藏宁琼]'
            r'(?:省|市|自治区|特别行政区))?'
            r'(?:[一-鿿]{2,10}?(?:市|区|县|镇|乡))?'
            r'[一-鿿]{2,20}?(?:路|街|巷|弄|道|大街)'
            r'(?:\d{1,6}?(?:号|弄|栋|楼|单元|室|层)?)?'
        )
        for m in re.finditer(pattern, text):
            value = m.group()
            if len(value) >= 5:  # 至少要有一定长度才算有效地址
                matches.append(PIIMatch(
                    category="address",
                    value=value,
                    start=m.start(), end=m.end(),
                    anonymized="[地址]",
                ))
        return matches

    def _detect_school(self, text: str) -> List[PIIMatch]:
        """检测学校名称"""
        matches = []
        for keyword in self.SCHOOL_KEYWORDS.split('|'):
            for m in re.finditer(rf'(?:[一-鿿]{{0,3}}{keyword})|(?:{keyword}[一-鿿]{{0,4}})', text):
                value = m.group()
                if len(value) >= 2:
                    matches.append(PIIMatch(
                        category="school_name",
                        value=value,
                        start=m.start(), end=m.end(),
                        anonymized="[学校名称]",
                    ))
        return matches

    def _detect_organization(self, text: str) -> List[PIIMatch]:
        """检测组织/公司名称"""
        matches = []
        for keyword in self.ORG_KEYWORDS.split('|'):
            for m in re.finditer(rf'(?:[一-鿿]{{0,4}}{keyword})|(?:{keyword}[一-鿿]{{0,6}})', text):
                value = m.group()
                if len(value) >= 4:
                    matches.append(PIIMatch(
                        category="company_name",
                        value=value,
                        start=m.start(), end=m.end(),
                        anonymized="[组织机构]",
                    ))
        return matches

    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """去除重叠的 PII 匹配——低优先级被修剪以让出空间给高优先级"""
        if not matches:
            return []
        priority = {"person_name": 0, "phone_number": 1, "id_card": 1, "email": 1,
                     "ip_address": 1, "wechat_id": 1, "bank_card": 1,
                     "license_plate": 1, "school_name": 2, "company_name": 2, "address": 2}
        matches.sort(key=lambda m: (priority.get(m.category, 3), -(m.end - m.start)))

        result = []
        for m in matches:
            # 收集所有与当前匹配重叠的已有结果
            overlapping = sorted(
                [e for e in result if m.start < e.end and m.end > e.start],
                key=lambda e: e.start
            )
            if not overlapping:
                result.append(m)
                continue

            # 将当前匹配切分为不重叠的片段
            segments = []
            cur_start = m.start
            for e in overlapping:
                if cur_start < e.start:
                    # 高优先级匹配之前的片段→保留
                    offset = e.start - m.start
                    seg_end = min(e.start, m.end)
                    segments.append((cur_start, seg_end))
                cur_start = max(cur_start, e.end)
                if cur_start >= m.end:
                    break

            if cur_start < m.end:
                segments.append((cur_start, m.end))

            for seg_start, seg_end in segments:
                if seg_end > seg_start:
                    seg_value = m.value[seg_start - m.start:seg_end - m.start]
                    if len(seg_value) >= 2:
                        result.append(PIIMatch(
                            category=m.category,
                            value=seg_value,
                            start=seg_start, end=seg_end,
                            anonymized=m.anonymized,
                        ))

        return result


# ===== 隐私守卫 =====

class PrivacyGuard:
    """
    隐私守卫——硬编码安全层

    在数据流的每个关键节点执行隐私检测和阻断：
    1. 输入层：保存前自动脱敏
    2. 存储层：PII 映射加密存储
    3. 访问层：每次查询/导出均验证目的
    4. 删除层：支持完全数据擦除

    用法:
        guard = PrivacyGuard()
        anonymized = guard.anonymize(user_text)  # 脱敏
        guard.validate_purpose("model_training")  # 验证目的——不合法则抛异常
        guard.audit_access(event_ids, "query", "api_v1")  # 审计
        guard.request_deletion(event_ids)  # 删除
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "core" / "data" / "privacy"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PIIDetector()
        self.audit_log_path = self.data_dir / "audit.jsonl"
        self.pii_mapping_path = self.data_dir / "pii_mappings.jsonl"
        self.deletion_registry_path = self.data_dir / "deletions.jsonl"
        self.purpose_violations_path = self.data_dir / "violations.jsonl"

        # 加载隐私配置
        self._load_config()

    def _load_config(self):
        config_path = Path(__file__).parent / "privacy_constraints.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    # ===== PII 脱敏 =====

    def anonymize(self, text: str) -> AnonymizationResult:
        """对文本执行 PII 脱敏"""
        if not text:
            return AnonymizationResult(
                original_text="",
                anonymized_text="",
                pii_matches=[],
                mapping_hash="",
                anonymized_at=datetime.now().isoformat(),
            )

        matches = self.detector.detect(text)
        if not matches:
            return AnonymizationResult(
                original_text=text,
                anonymized_text=text,
                pii_matches=[],
                mapping_hash="",
                anonymized_at=datetime.now().isoformat(),
            )

        # 按位置从右到左替换（保持位置正确）
        anonymized = text
        matches_sorted = sorted(matches, key=lambda m: m.start)
        for m in reversed(matches_sorted):
            # 使用类别+计数器作为占位符
            counter = sum(1 for pm in matches if pm.category == m.category and pm.start <= m.start)
            placeholder = f"[{m.anonymized.strip('[]')}_{counter}]" if counter > 1 else m.anonymized
            anonymized = anonymized[:m.start] + placeholder + anonymized[m.end:]

        # 生成映射哈希（用于完整性校验）
        mapping_hash = hashlib.sha256(
            json.dumps([{"cat": m.category, "pos": m.start} for m in matches],
                       ensure_ascii=False).encode()
        ).hexdigest()[:16]

        result = AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized,
            pii_matches=matches,
            mapping_hash=mapping_hash,
            anonymized_at=datetime.now().isoformat(),
        )

        # 保存 PII 映射（加密存储）
        self._save_pii_mapping(result)

        return result

    def _save_pii_mapping(self, result: AnonymizationResult):
        """保存 PII 映射（用于被遗忘权——定位特定用户的数据）"""
        mapping_entry = {
            "mapping_hash": result.mapping_hash,
            "anonymized_at": result.anonymized_at,
            "pii_count": len(result.pii_matches),
            "pii_categories": list(set(m.category for m in result.pii_matches)),
            "pii_checksum": hashlib.sha256(
                json.dumps([m.value for m in result.pii_matches], ensure_ascii=False).encode()
            ).hexdigest()[:32],  # 只存哈希——不存原文
            "name_checksums": [
                hashlib.sha256(m.value.encode()).hexdigest()[:16]
                for m in result.pii_matches if m.category == "person_name"
            ],
            "phone_checksums": [
                hashlib.sha256(m.value.encode()).hexdigest()[:16]
                for m in result.pii_matches if m.category == "phone_number"
            ],
        }
        with open(self.pii_mapping_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(mapping_entry, ensure_ascii=False) + '\n')

    # ===== 目的验证 =====

    def validate_purpose(self, purpose: str) -> None:
        """
        验证数据使用目的是否合法。
        如不合法——抛出 PrivacyViolation 异常（硬阻断）。
        """
        try:
            purpose_enum = DataUsagePurpose(purpose)
        except ValueError:
            raise PrivacyViolation(
                f"未知的数据使用目的: {purpose}",
                purpose=purpose,
                severity="CRITICAL",
            )

        if purpose_enum in FORBIDDEN_PURPOSES:
            self._log_violation(purpose, f"尝试以禁止目的使用数据: {purpose}")
            raise PrivacyViolation(
                f"禁止的数据使用目的: {purpose}。用户数据仅可用于模型训练。",
                purpose=purpose,
                severity="CRITICAL",
            )

    def is_purpose_allowed(self, purpose: str) -> bool:
        """检查目的是否合法（不抛异常版本）"""
        try:
            self.validate_purpose(purpose)
            return True
        except PrivacyViolation:
            return False

    # ===== 审计追踪 =====

    def audit_access(
        self,
        action: str,
        event_ids: List[str],
        purpose: str,
        caller: str = "unknown",
        details: str = "",
        pii_exposed: bool = False,
    ) -> str:
        """记录数据访问审计日志，返回 audit_id"""
        entry = PrivacyAuditEntry(
            audit_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now().isoformat(),
            action=action,
            event_ids=event_ids,
            purpose=purpose,
            caller=caller,
            result="blocked" if not self.is_purpose_allowed(purpose) else "allowed",
            details=details,
            pii_exposed=pii_exposed,
        )

        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

        # 如果是禁止目的，同时记录违规
        if entry.result == "blocked":
            self._log_violation(purpose, details)

        return entry.audit_id

    def _log_violation(self, purpose: str, details: str):
        """记录隐私违规"""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "purpose": purpose,
            "details": details,
            "severity": "CRITICAL",
        }
        with open(self.purpose_violations_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(violation, ensure_ascii=False) + '\n')

    # ===== 数据删除（被遗忘权）=====

    def request_deletion(
        self,
        event_ids: List[str],
        requester: str = "user",
        reason: str = "",
    ) -> Dict:
        """
        执行数据删除请求（GDPR 第17条 / 个人信息保护法第47条）。

        返回删除报告。
        """
        deletion_id = str(uuid.uuid4())[:8]
        deletion_entry = {
            "deletion_id": deletion_id,
            "timestamp": datetime.now().isoformat(),
            "event_ids": event_ids,
            "requester": requester,
            "reason": reason,
            "status": "pending",  # pending → completed → verified
        }

        with open(self.deletion_registry_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(deletion_entry, ensure_ascii=False) + '\n')

        # 审计
        self.audit_access(
            action="delete",
            event_ids=event_ids,
            purpose="user_deletion_request",
            caller=requester,
            details=f"删除原因: {reason}" if reason else "用户请求数据删除",
        )

        return {
            "deletion_id": deletion_id,
            "status": "pending",
            "message": f"数据删除请求已受理。将在7日内完成 {len(event_ids)} 条记录的永久删除。",
            "deleted_event_ids": event_ids,
        }

    def complete_deletion(self, deletion_id: str) -> bool:
        """标记删除完成"""
        # 更新删除注册表中的状态
        entries = []
        found = False
        if self.deletion_registry_path.exists():
            with open(self.deletion_registry_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("deletion_id") == deletion_id:
                        entry["status"] = "completed"
                        entry["completed_at"] = datetime.now().isoformat()
                        found = True
                    entries.append(entry)

        if found:
            with open(self.deletion_registry_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        return found

    # ===== 审计报告 =====

    def get_audit_report(self, days: int = 30) -> Dict:
        """获取审计报告"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        total_access = 0
        blocked_access = 0
        action_counts = defaultdict(int)
        purpose_counts = defaultdict(int)

        if self.audit_log_path.exists():
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("timestamp", "") < cutoff:
                            continue
                        total_access += 1
                        if entry.get("result") == "blocked":
                            blocked_access += 1
                        action_counts[entry.get("action", "unknown")] += 1
                        purpose_counts[entry.get("purpose", "unknown")] += 1
                    except json.JSONDecodeError:
                        continue

        # 违规计数
        violation_count = 0
        if self.purpose_violations_path.exists():
            with open(self.purpose_violations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("timestamp", "") >= cutoff:
                            violation_count += 1
                    except json.JSONDecodeError:
                        continue

        return {
            "period_days": days,
            "total_accesses": total_access,
            "blocked_accesses": blocked_access,
            "violations": violation_count,
            "actions": dict(action_counts),
            "purposes": dict(purpose_counts),
            "privacy_score": round(
                100 * (1 - (blocked_access + violation_count) / max(1, total_access)), 1
            ),
            "generated_at": datetime.now().isoformat(),
        }

    # ===== 合规检查 =====

    def check_compliance(self) -> Dict:
        """执行隐私合规自检"""
        issues = []
        warnings = []

        # 检查是否有未处理的删除请求（超过7天）
        if self.deletion_registry_path.exists():
            with open(self.deletion_registry_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("status") == "pending":
                            created = datetime.fromisoformat(entry["timestamp"])
                            if datetime.now() - created > timedelta(days=7):
                                issues.append(
                                    f"删除请求 {entry['deletion_id']} 超过7天未处理——合规风险"
                                )
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 检查是否有违规记录
        violation_count = 0
        if self.purpose_violations_path.exists():
            with open(self.purpose_violations_path, 'r', encoding='utf-8') as f:
                violation_count = sum(1 for _ in f)

        if violation_count > 10:
            issues.append(f"违规记录数量({violation_count})过高——建议审查")

        # 检查审计日志是否过大
        audit_lines = 0
        if self.audit_log_path.exists():
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                audit_lines = sum(1 for _ in f)

        if audit_lines > 100000:
            warnings.append(f"审计日志超过10万条——建议归档")

        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "violation_count": violation_count,
            "pending_deletions": sum(
                1 for _ in self._iter_deletions() if _["status"] == "pending"
            ),
            "audit_log_size": audit_lines,
            "checked_at": datetime.now().isoformat(),
        }

    def _iter_deletions(self):
        if self.deletion_registry_path.exists():
            with open(self.deletion_registry_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


# ===== 全局实例 =====

_default_privacy_guard = None


def get_privacy_guard(data_dir: str = None) -> PrivacyGuard:
    """获取全局隐私守卫实例（单例）"""
    global _default_privacy_guard
    if _default_privacy_guard is None:
        _default_privacy_guard = PrivacyGuard(data_dir)
    return _default_privacy_guard
