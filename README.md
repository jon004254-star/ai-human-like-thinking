# AI类人思维算法
人类是一种复杂的动物，一个自出生开始就对这个世界所有的一切都在疑惑。一些基本的生存由生物本能驱动，复杂的是情感需求，幼年时期的生存技能和情感与认知大部分都来自于父母和家庭成员与学校的教育。但是全部都是正确的吗？这里要打一个问号？当然这个世界没有标准答案，每一个人的经历与认知都是独一无二的。
但是每一个人都有无人相助，无人答疑解惑的时候，无助，空虚失去生活方向的时候。如何去解决这些问题，每个人其实都有自己方式与方法。如果ai与人类接触时能像人类一样去思考，更能理解人类的想法，分析人类的处境从而帮助人类更好的去完成工作，更好的去生活。当人类无助，悲伤，空虚，寂寞，孤单时有一个能理解你 开导你 给你的人生点亮一盏希望的灯，照亮你前进的方向的ai。你能与它分享你的喜怒哀乐，七情六欲，帮你排解空虚寂寞冷，还能保密。
现在是ai时代工具更多，信息更繁杂。如何确定哪些信息是你需要的，能快速获得的。这个事情就很难做到，哪怕是现在ai这么发达，在面对人类千奇百怪的想法与需求也显得笨的不行。毕竟人类的操作系统是宇宙慢长演化出来的，而且还在持续的进步当中。
所以如何让AI理解人类复杂的思维与想法。从人类底层本能出发，辅以理性分析，具体事件分析与演化推理，决策模拟概率，最终决策与时间戳（会不会为了长久利益放弃短期得利）等等去构建可计算的可用于ai的类人思维模型。

## 核心模型（暂时）

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
