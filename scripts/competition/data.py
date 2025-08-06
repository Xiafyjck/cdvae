from typing import List, Any
from pathlib import Path
import json

import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data, Batch

class CrystalDataset(Dataset):
    def __init__(
        self,
        mode: str = 'competition',
        root_path: str = None,
    ):
        super().__init__()
        self.mode = mode
        self.root_path = Path(root_path)

        if mode == 'competition':
            assert root_path is not None, "root_path must be provided for competition mode"
            self.composition_path = self.root_path / "composition.json"
            self.pattern_dir = self.root_path / "pattern"
            
            with open(self.composition_path, 'r') as f:
                self.compositions = json.load(f)
                self.sample_ids = list(self.compositions.keys())
        elif mode == 'train':
            raise NotImplementedError("Train mode is not implemented yet")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int):
        if self.mode == 'competition':
            return self._get_inference_item(idx)
        elif self.mode == 'train':
            raise NotImplementedError("Train mode is not implemented yet")
    
    def _get_inference_item(self, idx: int):
        sample_id = self.sample_ids[idx]
        
        # 读取对应的pattern文件
        pxrd_path = self.pattern_dir / f"{sample_id}.xy"
        assert pxrd_path.exists(), f"Pattern file {pxrd_path} does not exist"
        pxrd = np.loadtxt(pxrd_path, skiprows=1)
        pxrd_intensity = torch.FloatTensor(pxrd[:, 1])
        assert pxrd_intensity.size(0) == 11501, f"Expected 11501 intensity values, got {pxrd_intensity.size(0)}"
        
        # 获取组成
        niggli_comp_str, primitive_comp_str = self.compositions[sample_id]
        niggli_tensor = self._composition_to_aligned_tensor(niggli_comp_str)
        primitive_tensor = self._composition_to_aligned_tensor(primitive_comp_str, niggli_comp_str)

        sample = Data(
                sample_id = sample_id,
                niggli_tensor = niggli_tensor,
                primitive_tensor = primitive_tensor,
                pxrd_intensity = pxrd_intensity,
                num_niggli_atoms = len(niggli_tensor),
                num_primitive_atoms = len(primitive_tensor),
        )

        return sample

    def _composition_to_aligned_tensor(self, comp_str: str, reference_comp_str:
     str = None):
        """
        将化学组成转换为与参考组成对齐的张量
        
        Args:
            comp_str: 要转换的化学式
            reference_comp_str: 参考化学式（如Niggli），用于确定元素顺序和位置
        """
        comp = Composition(comp_str)

        if reference_comp_str is None:
            # 没有参考，直接转换
            atomic_numbers = [Element(el).Z
                              for el, amt in comp.items()
                              for _ in range(int(round(amt)))]
            return torch.LongTensor(atomic_numbers)

        # 有参考组成，需要对齐
        ref_comp = Composition(reference_comp_str)
        aligned_numbers = []

        # 按照参考组成的元素顺序处理
        for el, ref_amt in ref_comp.items():
            atomic_number = Element(el).Z
            current_amt = int(round(comp.get(el, 0)))  # 当前组成中该元素的数量
            ref_amt = int(round(ref_amt))

            # 添加实际数量的原子
            aligned_numbers.extend([atomic_number] * current_amt)
            # 用0填充不足的部分
            aligned_numbers.extend([0] * (ref_amt - current_amt))

        return torch.LongTensor(aligned_numbers)


def collate_sample(batch: List[Data]) -> Any:
    sample_ids = [sample.sample_id for sample in batch]
    niggli_tensors = [sample.niggli_tensor for sample in batch]
    primitive_tensors = [sample.primitive_tensor for sample in batch]
    pxrd_intensities = [sample.pxrd_intensity for sample in batch]
    num_niggli_atoms = [sample.num_niggli_atoms for sample in batch]
    num_primitive_atoms = [sample.num_primitive_atoms for sample in batch]

    # 拼接原子序数
    niggli_batch_tensor = torch.cat(niggli_tensors, dim=0)
    primitive_batch_tensor = torch.cat(primitive_tensors, dim=0)

    # 原子数量
    num_niggli_atoms_batch_tensor = torch.LongTensor(num_niggli_atoms)
    num_primitive_atoms_batch_tensor = torch.LongTensor(num_primitive_atoms)

    # 批次索引（哪个原子属于哪个样本）
    niggli_idx = torch.arange(len(batch)).repeat_interleave(num_niggli_atoms_batch_tensor)
    primitive_idx = torch.arange(len(batch)).repeat_interleave(num_primitive_atoms_batch_tensor)

    # PXRD 数据堆叠
    pxrd_batch = torch.stack(pxrd_intensities, dim=0)

    return Batch(
        # 样本信息
        sample_ids=sample_ids,
        batch_size=len(batch),

        # Niggli 表示
        niggli_atoms=niggli_batch_tensor,
        num_niggli_atoms=num_niggli_atoms_batch_tensor,
        niggli_batch=niggli_idx,

        # Primitive 表示
        primitive_atoms=primitive_batch_tensor,
        num_primitive_atoms=num_primitive_atoms_batch_tensor,
        primitive_batch=primitive_idx,

        # PXRD 数据
        pxrd_intensity=pxrd_batch,
    )


class CrystalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: CrystalDataset = None,
        batch_size: int = 32,
        num_workers: int = 2,
        collate_fn: callable = collate_sample,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def predict_dataloader(self):
        raise NotImplementedError("predict_dataloader is not implemented yet")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )