"""
数据加载器 —— 基础层，无业务逻辑。

加载本能数据库和思维演化数据。
所有引擎通过此加载器访问底层数据。
"""

import json
from typing import Optional
from pathlib import Path


class DataLoader:
    """加载本能数据库和思维演化数据"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.physiological = self._load_json("instincts/physiological.json")
        self.mental = self._load_json("instincts/mental.json")
        self.evolution = self._load_json("life_stages/thinking_evolution.json")

    def _load_json(self, relative_path: str) -> dict:
        path = self.data_dir / relative_path
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_instinct(self, name_en: str) -> Optional[dict]:
        """获取指定本能的数据"""
        for db in [self.physiological, self.mental]:
            if name_en in db.get("instincts", {}):
                return db["instincts"][name_en]
        return None

    def get_all_instincts(self) -> dict:
        """获取所有本能数据"""
        all_instincts = {}
        all_instincts.update(self.physiological.get("instincts", {}))
        all_instincts.update(self.mental.get("instincts", {}))
        return all_instincts

    def get_stage(self, age: float) -> dict:
        """根据年龄获取对应的思维阶段"""
        stages = self.evolution["stages"]
        stage_map = [
            ("infant_toddler", 0, 3),
            ("early_childhood", 3, 7),
            ("middle_childhood", 7, 12),
            ("adolescence", 12, 18),
            ("young_adult", 18, 25),
            ("early_adulthood", 25, 40),
            ("middle_age", 40, 60),
            ("late_adulthood", 60, None),
        ]
        for stage_name, min_age, max_age in stage_map:
            if min_age <= age and (max_age is None or age < max_age):
                return stages[stage_name]
        return stages["late_adulthood"]
