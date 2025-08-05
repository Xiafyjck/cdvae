#!/usr/bin/env python3
"""
比赛提交文件生成脚本
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from .data_utils import (
    load_competition_data, 
    get_composition_statistics
)
from .baseline_models import (
    RandomBaseline,
    CompositionMatchBaseline
)


def create_submission_csv(structures: Dict, cfg: DictConfig):
    """
    创建submission.csv文件
    
    Args:
        structures: {sample_id: Structure}
        cfg: 配置
    """
    rows = []
    
    # 获取ID前缀
    sample_ids = list(structures.keys())
    if sample_ids:
        prefix = sample_ids[0].split('-')[0]  # 获取 'A' 或 'B'
    else:
        raise ValueError("No structures generated!")
    
    print(f"Generating submission for {prefix} split with {len(structures)} structures")
    
    # 生成CIF文件
    for sample_id in tqdm(sorted(structures.keys()), desc="Converting to CIF"):
        structure = structures[sample_id]
        try:
            cif_string = structure.to(fmt="cif")
            rows.append([sample_id, cif_string])
        except Exception as e:
            print(f"Error converting {sample_id} to CIF: {e}")
            # 使用空CIF作为fallback
            rows.append([sample_id, ""])
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows, columns=["ID", "cif"])
    
    # 保存到输出路径
    output_path = Path(cfg.competition.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    submission_path = output_path / cfg.competition.output.submission_file
    df.to_csv(submission_path, index=False)
    
    print(f"Submission saved to: {submission_path}")
    print(f"Total entries: {len(df)}")
    
    # 验证文件格式
    if len(df) != len(structures):
        print(f"WARNING: Expected {len(structures)} entries, but got {len(df)}")
    
    # 检查是否有空的CIF
    empty_cifs = df[df['cif'] == ''].shape[0]
    if empty_cifs > 0:
        print(f"WARNING: {empty_cifs} entries have empty CIF strings")


def select_generator(cfg: DictConfig):
    """
    根据配置选择生成器
    """
    strategy = cfg.competition.generation.strategy
    
    if strategy == "random_sample":
        return RandomBaseline(cfg)
    elif strategy == "composition_match":
        return CompositionMatchBaseline(cfg)
    else:
        raise ValueError(f"Unknown generation strategy: {strategy}")


@hydra.main(version_base=None, config_path="../../conf", config_name="competition/default")
def main(cfg: DictConfig):
    """主函数"""
    
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # 1. 加载比赛数据
    print("Loading competition data...")
    compositions = load_competition_data(cfg)
    print(f"Loaded {len(compositions)} samples")
    
    # 打印数据统计信息
    stats = get_composition_statistics(compositions)
    print(f"Element types: {len(stats['element_types'])} unique elements")
    print(f"Atoms per structure: {min(stats['num_atoms_distribution'])} - {max(stats['num_atoms_distribution'])}")
    
    # 2. 选择生成器
    print(f"\nUsing generation strategy: {cfg.competition.generation.strategy}")
    generator = select_generator(cfg)
    
    # 3. 生成结构
    print("\nGenerating structures...")
    structures = generator.generate(compositions)
    
    # 4. 创建提交文件
    print("\nCreating submission file...")
    create_submission_csv(structures, cfg)
    
    print("\nDone!")


if __name__ == "__main__":
    main()