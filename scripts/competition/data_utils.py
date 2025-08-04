import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import re


def load_competition_data(cfg) -> Dict:
    """
    加载比赛数据，包括composition和XRD pattern
    
    Returns:
        Dict: {sample_id: {"composition": [niggli, primitive], "pattern_path": str}}
    """
    data_path = Path(cfg.competition.data_path)
    split = cfg.competition.dataset.split
    
    # 加载composition
    comp_path = data_path / split / cfg.competition.dataset.composition_file
    with open(comp_path, 'r') as f:
        compositions = json.load(f)
    
    # 构建数据字典
    data = {}
    pattern_dir = data_path / split / cfg.competition.dataset.pattern_dir
    
    for sample_id, comp_info in compositions.items():
        data[sample_id] = {
            "composition": comp_info["composition"],  # [niggli, primitive]
            "pattern_path": str(pattern_dir / f"{sample_id}.xy")
        }
    
    return data


def parse_composition(comp_str: str) -> Dict[str, int]:
    """
    解析composition字符串
    
    Args:
        comp_str: 如 "Sr4 Be2 Re2 N8"
    
    Returns:
        Dict: {"Sr": 4, "Be": 2, "Re": 2, "N": 8}
    """
    elements = {}
    # 使用正则表达式匹配元素和数量
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, comp_str)
    
    for element, count in matches:
        if element:  # 确保不是空字符串
            count = int(count) if count else 1
            elements[element] = count
    
    return elements


def composition_to_formula(elements: Dict[str, int]) -> str:
    """
    将元素字典转换回formula字符串
    
    Args:
        elements: {"Sr": 4, "Be": 2}
    
    Returns:
        str: "Sr4 Be2"
    """
    formula_parts = []
    for element, count in sorted(elements.items()):
        if count == 1:
            formula_parts.append(element)
        else:
            formula_parts.append(f"{element}{count}")
    return " ".join(formula_parts)


def load_xrd_pattern(pattern_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载XRD pattern (.xy格式)
    
    Args:
        pattern_path: .xy文件路径
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (2theta角度, 强度)
    """
    data = np.loadtxt(pattern_path, skiprows=1)  # 跳过header
    two_theta = data[:, 0]
    intensity = data[:, 1]
    return two_theta, intensity


def get_element_types(compositions: Dict) -> List[str]:
    """
    获取数据集中所有出现的元素类型
    
    Args:
        compositions: 比赛数据字典
    
    Returns:
        List[str]: 元素列表
    """
    all_elements = set()
    
    for sample_data in compositions.values():
        # 分析两种composition
        for comp_str in sample_data["composition"]:
            elements = parse_composition(comp_str)
            all_elements.update(elements.keys())
    
    return sorted(list(all_elements))


def get_composition_statistics(compositions: Dict) -> Dict:
    """
    统计composition信息
    
    Returns:
        Dict: 包含元素种类数、原子总数等统计信息
    """
    stats = {
        "num_samples": len(compositions),
        "element_types": get_element_types(compositions),
        "num_atoms_distribution": [],
        "num_elements_distribution": []
    }
    
    for sample_data in compositions.values():
        # 使用primitive cell的composition进行统计
        comp_str = sample_data["composition"][1]  # primitive cell
        elements = parse_composition(comp_str)
        
        num_atoms = sum(elements.values())
        num_elements = len(elements)
        
        stats["num_atoms_distribution"].append(num_atoms)
        stats["num_elements_distribution"].append(num_elements)
    
    return stats