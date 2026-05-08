"""
验证场景测试运行器

运行方式: python -m src.tests.test_runner
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.human_thinking_engine import (
    HumanThinkingEngine, PersonProfile, EventContext
)


def load_scenarios() -> list:
    """加载验证场景"""
    path = Path(__file__).parent / "validation_scenarios.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["scenarios"]


def run_scenario(engine: HumanThinkingEngine, scenario: dict) -> dict:
    """运行单个验证场景"""
    p = scenario["person"]
    e = scenario["event"]

    person = PersonProfile(
        age=p["age"],
        health_status=p.get("health_status", "normal"),
        emotional_state=p.get("emotional_state", "neutral"),
        social_context=p.get("social_context", "alone"),
        recent_events=p.get("recent_events", []),
    )

    event = EventContext(
        event_type=e.get("event_type", "unknown"),
        event_description=e.get("event_description", ""),
        threat_level=e.get("threat_level", 0.0),
        social_threat_level=e.get("social_threat_level", 0.0),
        resource_scarcity=e.get("resource_scarcity", 0.0),
        social_visibility=e.get("social_visibility", 0.0),
        time_pressure=e.get("time_pressure", 0.0),
        anonymity=e.get("anonymity", 0.0),
    )

    result = engine.judge(person, event)
    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "result": result
    }


def validate_result(result: dict, expected: dict) -> list:
    """校验结果是否符合预期"""
    issues = []
    judgment = result["result"]
    expected_instincts = expected.get("expected_instincts", {})

    for name_en, exp in expected_instincts.items():
        # 查找匹配的本能（可能用中文名或英文名索引）
        found = False
        for key, act in judgment.instinct_activations.items():
            if name_en in key or name_en in act.instinct_name:
                found = True
                if "min_activation" in exp:
                    if act.current_activation < exp["min_activation"]:
                        issues.append(
                            f"  ⚠ {act.instinct_name}: activation={act.current_activation:.3f} "
                            f"< expected min={exp['min_activation']}"
                        )
                    else:
                        issues.append(
                            f"  ✓ {act.instinct_name}: activation={act.current_activation:.3f} "
                            f"(≥ {exp['min_activation']})"
                        )
                if "state" in exp and act.state != exp["state"]:
                    issues.append(
                        f"  ⚠ {act.instinct_name}: state={act.state} != expected {exp['state']}"
                    )
                break
        if not found:
            issues.append(f"  ✗ {name_en}: not found in activations")

    return issues


def main():
    engine = HumanThinkingEngine()
    scenarios = load_scenarios()

    print("=" * 70)
    print("人类思维引擎 - 验证场景测试")
    print("=" * 70)

    total = len(scenarios)
    passed = 0
    warnings = 0
    failed = 0

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"测试 {scenario['id']}: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        if "reference" in scenario:
            print(f"参考: {scenario['reference']}")
        print(f"{'─' * 70}")

        result = run_scenario(engine, scenario)
        judgment = result["result"]

        print(f"人物: 年龄={judgment.person_profile.age}, "
              f"健康={judgment.person_profile.health_status}, "
              f"情绪={judgment.person_profile.emotional_state}")
        print(f"社会调制系数: {judgment.social_modulation_coefficient:.2f}")
        print(f"主导驱动: {', '.join(judgment.dominant_drivers)}")
        print(f"预测行为: {judgment.predicted_behavior_pattern}")
        print(f"预测情绪: {judgment.predicted_emotional_response}")
        print(f"置信度: {judgment.confidence:.2f}")
        print(f"安全审核: {'✓ 通过' if judgment.safety_check_passed else '✗ 未通过'}")

        # 显示所有激活的本能
        print("\n本能激活度:")
        sorted_acts = sorted(
            judgment.instinct_activations.items(),
            key=lambda x: x[1].current_activation, reverse=True
        )
        for name_en, act in sorted_acts:
            if act.current_activation > 0.05:
                bar = "█" * int(act.current_activation * 20)
                print(f"  {act.instinct_name:8s} [{act.state:7s}] {bar} {act.current_activation:.3f}")

        if judgment.notes:
            print("\n备注:")
            for note in judgment.notes:
                print(f"  • {note}")

        # 校验
        print("\n期望校验:")
        if "expected_instincts" in scenario:
            issues = validate_result(result, scenario)
            for issue in issues:
                print(issue)
            has_warnings = any("⚠" in i for i in issues)
            has_errors = any("✗" in i for i in issues)
            if has_errors:
                failed += 1
            elif has_warnings:
                warnings += 1
                print("  → 部分校验未通过（需要调整参数）")
            else:
                passed += 1
                print("  → 校验通过")
        else:
            print("  (无期望值定义)")

        # 安全注意事项
        if not judgment.safety_check_passed:
            print(f"\n⚠️ 安全审核未通过！请检查安全警告。")

    print(f"\n{'=' * 70}")
    print(f"测试汇总: {total} 场景 | ✓ 通过: {passed} | ⚠ 部分: {warnings} | ✗ 失败: {failed}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
