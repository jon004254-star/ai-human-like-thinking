# AI类人思维算法

> 让 AI 理解人类复杂的思维与想法。从人类底层本能出发，构建可计算的思维模型。

## 核心模型

```
人类行为决策 = 本能信号 × 社会调制系数 + 理性计算
```

- **本能信号**：来自 12 条生理本能 + 18 条精神本能的底层驱动力
- **社会调制系数**：文化、规范、教育对本能的压制/引导因子（0.0 ~ 1.0）
- **理性计算**：前额叶主导的目标导向推理

**成年人日常行为中，社会调制系数约 0.80 ~ 0.95——本能是微弱的背景信号。** 只有在生理剥夺、急性威胁、情绪劫持或去个体化时，本能才会裸露。

## 项目结构

```
├── docs/framework/                    # 框架文档（理论层）
│   ├── 00-overview.md                 #   总纲：双层模型、学术引用索引
│   ├── 01-physiological-instincts.md  #   生理本能 12 条
│   └── 02-mental-instincts.md         #   精神本能 18 条
│
├── src/
│   ├── data/
│   │   ├── instincts/
│   │   │   ├── physiological.json     #   生理本能结构化数据库
│   │   │   └── mental.json            #   精神本能结构化数据库
│   │   └── life_stages/
│   │       └── thinking_evolution.json #  思维演化——8 个年龄阶段
│   │
│   ├── engine/
│   │   └── human_thinking_engine.py   #   核心判定引擎
│   │
│   ├── tests/
│   │   ├── validation_scenarios.json  #   验证场景集（8 个经典场景）
│   │   └── test_runner.py             #   测试运行器
│   │
│   └── safety/
│       └── safety_constraints.json    #   安全约束（硬编码，不可绕过）
```

## 快速开始

```python
from src.engine import HumanThinkingEngine, create_person, create_event

engine = HumanThinkingEngine()

# 构建被判定者画像
person = create_person(age=16, social="classroom")

# 构建事件情境
event = create_event(
    "social_threat",
    "在全班同学面前被老师严厉批评",
    social_threat=0.75,
    social_visibility=0.95
)

# 执行判定
result = engine.judge(person, event)

print(f"社会调制系数: {result.social_modulation_coefficient:.2f}")
print(f"主导驱动: {result.dominant_drivers}")
print(f"预测情绪: {result.predicted_emotional_response}")
print(f"安全审核: {result.safety_check_passed}")
```

## 安全约束

本项目包含硬编码的安全约束层，不可被任何上层逻辑绕过：

- **禁止**使用本能知识操纵、诱导或欺骗人类
- **禁止**针对弱势群体（儿童/老人/精神障碍者）使用本能分析
- **禁止**设计成瘾性产品或去个体化操纵环境
- 所有判定结果自动经过安全审计

## 参考著作

Damasio《笛卡尔的错误》、Panksepp《情感神经科学》、Kahneman《思考快与慢》、Frankl《活出生命的意义》、Becker《拒斥死亡》、Sapolsky《为什么斑马不得胃溃疡》、LeDoux《情绪脑》、Festinger 认知失调理论、Baumeister 归属理论 等。
