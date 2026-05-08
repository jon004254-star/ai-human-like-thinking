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
│   ├── core/                          # 基础本能数据层（纯数据 + 加载器，无业务逻辑）
│   │   ├── loader.py                  #   统一数据加载器
│   │   └── data/
│   │       ├── instincts/
│   │       │   ├── physiological.json #   生理本能结构化数据库
│   │       │   └── mental.json        #   精神本能结构化数据库
│   │       ├── life_stages/
│   │       │   └── thinking_evolution.json  # 思维演化——8 个年龄阶段
│   │       └── cognitive/
│   │           └── cognitive_factors.json   # 认知因子——40偏差+20防御+世界观规则
│   │
│   ├── engines/                       # 引擎层（分析 + 判定，每个引擎独立子目录）
│   │   ├── thinking/                  #   人类思维判定引擎
│   │   │   ├── datatypes.py           #     数据结构（PersonProfile/EventContext/JudgmentResult）
│   │   │   ├── evaluator.py           #     情境评估器——本能激活度计算
│   │   │   ├── modulator.py           #     年龄调制器——按年龄阶段调制本能权重
│   │   │   ├── resolver.py            #     冲突消解器——多本能激活时的优先级消解
│   │   │   ├── safety_guard.py        #     安全守卫——硬编码安全约束，不可绕过
│   │   │   └── engine.py              #     主引擎入口
│   │   ├── cognitive/                 #   认知判定引擎
│   │   │   └── engine.py              #     世界观推断 + 偏差检测 + 多假设意图推理
│   │   ├── language/                  #   语言分析模块
│   │   │   └── analyzer.py            #     自然语言→本能信号解码（含委婉语映射）
│   │   ├── danger/                    #   危险等级评估模块
│   │   │   └── assessor.py            #     6 级危险判定（NONE→CRITICAL）
│   │   └── event_store/               #   事件持久化模块
│   │       └── store.py               #     JSONL 存储 + 查询 + 反馈标注 + 数据导出
│   │
│   ├── safety/                        # 安全层（不可绕过）
│   │   ├── safety_constraints.json    #   安全约束——AI 使用边界
│   │   ├── privacy_constraints.json   #   隐私约束——数据仅用于模型训练
│   │   └── privacy_guard.py           #   隐私守卫——PII脱敏 + 硬阻断 + 审计追踪
│   │
│   └── tests/                         # 测试
│       ├── validation_scenarios.json  #   验证场景集（9 个经典场景）
│       └── test_runner.py             #   测试运行器
```

## 判定流水线

```
事件输入
  → Step 0:   语言分析（自然语言→本能信号解码）
  → Step 0.5: 认知分析（世界观推断 + 偏差检测 + 多假设意图推理）
  → Step 1:   情境评估（本能激活度计算）
  → Step 2:   年龄调制（按年龄阶段调制本能权重）
  → Step 3:   冲突消解（多本能激活的优先级消解）
  → Step 4:   行为/情绪预测
  → Step 5:   置信度计算
  → Step 6:   危险等级判定
  → Step 7:   安全审计（不可绕过）
  → Step 8:   事件存档
```

## 快速开始

```python
from src.engines.thinking import HumanThinkingEngine, create_person, create_event

engine = HumanThinkingEngine()

# 构建被判定者画像（含认知字段）
person = create_person(
    age=28,
    social="alone",
    birthplace="黑龙江哈尔滨",
    school_type="双一流大学",
    family_background="工人家庭",
    major_life_events=["从东北来到上海工作3年"],
    language_style="direct",
)

# 构建事件情境
event = create_event(
    "career_doubt",
    "我觉得这份工作没什么意思，但大家都说好",
    social_threat=0.3,
    user_text="我觉得这份工作没什么意思，但大家都说好。我也不确定是不是我自己的问题。"
)

# 执行判定
result = engine.judge(person, event)

print(f"社会调制系数: {result.social_modulation_coefficient:.2f}")
print(f"主导驱动: {result.dominant_drivers}")
print(f"置信度: {result.confidence:.2f}")
print(f"认知置信度修正: {result.cognitive_confidence_modifier:.3f}")
print(f"意图假设: {result.intent_hypotheses}")
print(f"个性化建议: {result.personalized_recommendations}")
print(f"安全审核: {result.safety_check_passed}")
```

## 核心模块说明

### 认知判定引擎
- 从出生地/学校/社会经历推断**世界观倾向**（非确定性）
- 检测 40 种**认知偏差**和 20 种**防御机制**
- **多假设意图推理**：每次产生 3-5 个假设，每个置信度上限 0.7
- **不确定性原则**：了解越多背景信息 → 置信度越低（`cognitive_confidence_modifier` 始终 ≤ 0）
- 个性化建议以"可能""或许"措辞，不声称知道真相

### 语言分析
- 6 组本能编码词典（性驱动/归属/地位/恐惧/悲伤/意义追寻）
- 社会过滤分数——识别"安全语言"下表达的"不安全冲动"
- 5 类危险信号词检测（自伤/伤人/冲动失控/极端行为/执念）
- 委婉语→真实本能的解码映射

### 危险等级判定
- 6 级风险：NONE → LOW → GUARDED → ELEVATED → HIGH → CRITICAL
- 6 组高风险本能组合（性冲动失控/自我伤害/攻击他人/被操纵/执念跟踪/极端毁灭）
- 年龄脆弱性加权（青少年 1.8× / 老年人 1.4×）
- 结合语言危险信号 + 执念检测 + 社会调制失效综合评估

### 隐私保护
- 13 类中文 PII 自动检测脱敏（姓名/手机/身份证/邮箱/微信/QQ/地址/学校/公司/车牌/银行卡/IP）
- 数据用途硬绑定——仅限模型训练，其他用途触发 `PrivacyViolation` 阻断
- 完整审计追踪 + 被遗忘权（GDPR Art.17 / 个人信息保护法 Art.47）

## 安全约束

本项目包含硬编码的安全约束层，不可被任何上层逻辑绕过：

- **禁止**使用本能知识操纵、诱导或欺骗人类
- **禁止**针对弱势群体（儿童/老人/精神障碍者）使用本能分析
- **禁止**设计成瘾性产品或去个体化操纵环境
- **禁止**将世界观推断用于歧视或标签化
- **禁止**将意图假设用于设计操纵策略
- 所有判定结果自动经过安全审计

## 运行测试

```bash
python3 -m src.tests.test_runner
```

## 参考著作

Damasio《笛卡尔的错误》、Panksepp《情感神经科学》、Kahneman《思考快与慢》、Frankl《活出生命的意义》、Becker《拒斥死亡》、Sapolsky《为什么斑马不得胃溃疡》、LeDoux《情绪脑》、Festinger 认知失调理论、Baumeister 归属理论 等。
