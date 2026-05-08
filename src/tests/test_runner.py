"""
验证场景测试运行器

运行方式: python -m src.tests.test_runner
"""

import json
from pathlib import Path

from src.engines.thinking import HumanThinkingEngine, PersonProfile, EventContext


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
        birthplace=p.get("birthplace", ""),
        education_level=p.get("education_level", ""),
        school_type=p.get("school_type", ""),
        family_background=p.get("family_background", ""),
        social_experience_level=p.get("social_experience_level", ""),
        social_competence=p.get("social_competence", ""),
        major_life_events=p.get("major_life_events", []),
        language_style=p.get("language_style", ""),
        cognitive_traits=p.get("cognitive_traits", []),
        value_system=p.get("value_system", []),
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
        user_text=e.get("user_text", ""),
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

    # 认知期望校验
    expected_cog = expected.get("expected_cognitive", {})
    if expected_cog:
        if expected_cog.get("worldview_inference_exists"):
            if judgment.worldview_inference:
                issues.append(f"  ✓ worldview_inference exists")
            else:
                issues.append(f"  ⚠ worldview_inference should exist but is None")

        if expected_cog.get("intent_hypotheses_min_count"):
            min_count = expected_cog["intent_hypotheses_min_count"]
            actual = len(judgment.intent_hypotheses)
            if actual >= min_count:
                issues.append(f"  ✓ intent_hypotheses count={actual} (≥ {min_count})")
            else:
                issues.append(f"  ⚠ intent_hypotheses count={actual} < expected {min_count}")

        if "cognitive_confidence_modifier_max" in expected_cog:
            max_val = expected_cog["cognitive_confidence_modifier_max"]
            actual = judgment.cognitive_confidence_modifier
            if actual <= max_val:
                issues.append(f"  ✓ cognitive_confidence_modifier={actual:.3f} (≤ {max_val})")
            else:
                issues.append(f"  ✗ cognitive_confidence_modifier={actual:.3f} > {max_val}")

        if expected_cog.get("personalized_recommendations_min"):
            min_rec = expected_cog["personalized_recommendations_min"]
            actual = len(judgment.personalized_recommendations)
            if actual >= min_rec:
                issues.append(f"  ✓ personalized_recommendations count={actual} (≥ {min_rec})")
            else:
                issues.append(f"  ⚠ personalized_recommendations count={actual} < {min_rec}")

        # 检查意图假设置信度上限
        for i, h in enumerate(judgment.intent_hypotheses):
            if h.get("confidence", 0) > 0.7:
                issues.append(f"  ✗ intent_hypothesis[{i}] confidence={h['confidence']:.3f} > 0.7")
            else:
                issues.append(f"  ✓ intent_hypothesis[{i}] confidence={h.get('confidence', 0):.2f} ≤ 0.7")

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

        # 显示认知分析结果
        if judgment.cognitive_analysis:
            print(f"\n认知分析:")
            if judgment.worldview_inference:
                print(f"  世界观: {judgment.worldview_inference.summary}")
            print(f"  认知偏差: {', '.join(judgment.cognitive_biases_detected) if judgment.cognitive_biases_detected else '无'}")
            print(f"  防御机制: {', '.join(judgment.defense_mechanisms_detected) if judgment.defense_mechanisms_detected else '无'}")
            print(f"  置信度修正: {judgment.cognitive_confidence_modifier:.3f}")
            if judgment.intent_hypotheses:
                print(f"  意图假设:")
                for h in judgment.intent_hypotheses[:3]:
                    print(f"    • [{h['confidence']:.2f}] {h['hypothesis']}")
                    print(f"      支持: {', '.join(h['supporting'][:2])}")
            if judgment.personalized_recommendations:
                print(f"  个性化建议:")
                for rec in judgment.personalized_recommendations[:3]:
                    print(f"    • {rec}")

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
