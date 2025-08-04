import torch
import numpy as np
from typing import Dict, List, Optional
from pymatgen.core import Structure, Lattice, Element
from cdvae.pl_modules.model import CDVAE
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from .data_utils import parse_composition


class BaselineGenerator:
    """基础生成器类"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.competition.model.device if torch.cuda.is_available() else "cpu")
        
    def generate(self, compositions: Dict) -> Dict[str, Structure]:
        """生成结构的接口方法"""
        raise NotImplementedError
        

class RandomBaseline(BaselineGenerator):
    """
    随机生成baseline
    使用预训练CDVAE随机生成结构，然后匹配composition
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self._load_model()
        
    def _load_model(self) -> CDVAE:
        """加载预训练的CDVAE模型"""
        checkpoint_path = Path(self.cfg.competition.model.checkpoint_path)
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 从checkpoint重建模型
        model = CDVAE.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False  # 允许部分参数不匹配
        )
        model.to(self.device)
        model.eval()
        
        return model
    
    def generate(self, compositions: Dict) -> Dict[str, Structure]:
        """
        为每个composition生成结构
        """
        structures = {}
        
        with torch.no_grad():
            for sample_id, sample_data in tqdm(compositions.items(), desc="Generating structures"):
                # 获取目标composition
                target_comp = parse_composition(sample_data["composition"][1])  # 使用primitive cell
                num_atoms = sum(target_comp.values())
                
                # 生成多个候选结构
                best_structure = None
                
                for _ in range(self.cfg.competition.generation.num_samples):
                    try:
                        # 随机采样latent code
                        z = torch.randn(1, self.model.hparams.latent_dim).to(self.device)
                        
                        # 生成结构
                        structure = self._decode_structure(z, num_atoms, target_comp)
                        
                        if structure is not None:
                            best_structure = structure
                            break  # 找到合适的结构就停止
                            
                    except Exception as e:
                        print(f"Error generating structure for {sample_id}: {e}")
                        continue
                
                # 如果没有生成成功，使用默认结构
                if best_structure is None:
                    best_structure = self._create_default_structure(target_comp)
                
                structures[sample_id] = best_structure
                
        return structures
    
    def _decode_structure(self, z: torch.Tensor, num_atoms: int, target_comp: Dict[str, int]) -> Optional[Structure]:
        """
        从latent code解码结构
        """
        # 这里需要根据CDVAE的具体实现来调整
        # 暂时返回None，实际实现需要调用model的decoder
        return None
    
    def _create_default_structure(self, composition: Dict[str, int]) -> Structure:
        """
        创建默认的立方晶格结构
        """
        # 计算总原子数
        total_atoms = sum(composition.values())
        
        # 创建简单立方晶格
        a = 5.0  # 默认晶格常数
        lattice = Lattice.cubic(a)
        
        # 构建原子列表
        species = []
        coords = []
        
        atom_index = 0
        for element, count in composition.items():
            for _ in range(count):
                species.append(Element(element))
                # 在单位晶胞内均匀分布
                x = (atom_index % 2) * 0.5
                y = ((atom_index // 2) % 2) * 0.5
                z = ((atom_index // 4) % 2) * 0.5
                coords.append([x, y, z])
                atom_index += 1
        
        return Structure(lattice, species, coords)


class CompositionMatchBaseline(BaselineGenerator):
    """
    基于composition匹配的baseline
    从训练数据中找相似composition的结构作为模板
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.template_structures = self._load_template_structures()
        
    def _load_template_structures(self) -> Dict:
        """加载模板结构库"""
        # 这里应该从训练集加载结构
        # 暂时返回空字典
        return {}
    
    def generate(self, compositions: Dict) -> Dict[str, Structure]:
        """基于composition相似度生成结构"""
        structures = {}
        
        for sample_id, sample_data in tqdm(compositions.items(), desc="Matching structures"):
            target_comp = parse_composition(sample_data["composition"][1])
            
            # 查找最相似的模板
            best_match = self._find_best_match(target_comp)
            
            if best_match is not None:
                # 调整模板以匹配目标composition
                structure = self._adapt_structure(best_match, target_comp)
            else:
                # 使用默认结构
                structure = RandomBaseline._create_default_structure(self, target_comp)
            
            structures[sample_id] = structure
            
        return structures
    
    def _find_best_match(self, target_comp: Dict[str, int]) -> Optional[Structure]:
        """查找最匹配的模板结构"""
        # TODO: 实现composition相似度计算
        return None
    
    def _adapt_structure(self, template: Structure, target_comp: Dict[str, int]) -> Structure:
        """调整模板结构以匹配目标composition"""
        # TODO: 实现结构调整逻辑
        return template