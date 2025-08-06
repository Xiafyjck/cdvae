"""
Competition evaluation script for CDVAE
简化版本：去除批处理，添加composition约束，用于AI4S Cup比赛
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from pymatgen.core import Composition, Element
import pandas as pd
from pymatgen.core import Structure, Lattice

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from eval_utils import load_model


# ========== 硬编码配置 ==========
DATA_PATH = "/home/ma-user/work/cdvae/docs/比赛用例/data/A"
MODEL_PATH = "/home/ma-user/work/cdvae/outputs/mp20epochs100/08-06-18-09-53/outputs/mp20epochs100/08-06-18-09-53"
DRY_RUN = True  # 设为True时只处理前10个样本

# Langevin dynamics参数
LD_KWARGS = SimpleNamespace(
    n_step_each=10,      # 每个噪声水平的步数
    step_lr=1e-4,         # 步长
    min_sigma=0,          # 最小噪声水平
    save_traj=True,      # 不保存轨迹（节省内存）
    disable_bar=True      # 禁用进度条（避免嵌套进度条）
)

# 生成参数
NUM_CANDIDATES = 1  # 每个样本生成几个候选（先设为1，简化流程）


def parse_composition(comp_str):
    """
    将组成字符串转换为原子数量和原子类型
    
    输入: "Sr4 Be2 Re2 N8"
    输出: 
        num_atoms: tensor([16])  # 总原子数
        atom_types: tensor([38,38,38,38,4,4,75,75,7,7,7,7,7,7,7,7])  # 原子序数列表
    """
    comp = Composition(comp_str)
    atom_list = []
    
    for element, count in comp.items():
        atomic_num = Element(element).Z
        atom_list.extend([atomic_num] * int(count))
    
    num_atoms = torch.tensor([len(atom_list)])
    atom_types = torch.tensor(atom_list)
    
    return num_atoms, atom_types


def generation_for_competition(model, compositions_dict):
    """
    为比赛生成晶体结构
    
    Args:
        model: 加载好的CDVAE模型
        compositions_dict: 组成信息字典 {"A-1": {"composition": [...]}, ...}
    
    Returns:
        包含所有生成结构的字典
    """
    sample_ids = list(compositions_dict.keys())
    
    # DRY_RUN模式：只处理前10个
    if DRY_RUN:
        sample_ids = sample_ids[:10]
        print(f"DRY_RUN模式：只处理前{len(sample_ids)}个样本")
    
    # 准备输出容器
    all_frac_coords = []  # 所有样本的分数坐标
    all_num_atoms = []     # 所有样本的原子数
    all_atom_types = []    # 所有样本的原子类型
    all_lengths = []       # 所有样本的晶格长度
    all_angles = []        # 所有样本的晶格角度
    
    # 逐个处理每个样本
    for idx, sample_id in enumerate(tqdm(sample_ids, desc="生成晶体结构")):
        # 1. 解析组成信息
        comp_info = compositions_dict[sample_id]
        # 使用niggli reduced cell的组成（第一个）
        comp_str = comp_info["composition"][0]
        
        # 将组成转换为张量
        # num_atoms: [1] 例如 [16]
        # atom_types: [16] 例如 [38,38,38,38,4,4,75,75,7,7,7,7,7,7,7,7]
        gt_num_atoms, gt_atom_types = parse_composition(comp_str)
        
        if torch.cuda.is_available():
            gt_num_atoms = gt_num_atoms.cuda()
            gt_atom_types = gt_atom_types.cuda()
        
        # 2. 生成潜在向量z
        # z: [1, hidden_dim] 例如 [1, 128]
        z = torch.randn(1, model.hparams.hidden_dim, device=model.device)
        
        # 3. 条件生成：使用组成约束
        outputs = model.langevin_dynamics(
            z, 
            LD_KWARGS,
            gt_num_atoms=gt_num_atoms,   # 强制使用给定的原子数
            gt_atom_types=gt_atom_types  # 强制使用给定的原子类型
        )
        
        # 4. 提取生成的结构
        # frac_coords: [num_atoms, 3] 例如 [16, 3]
        # num_atoms: [1] 例如 [16]
        # atom_types: [num_atoms] 例如 [38,38,38,38,4,4,75,75,7,7,7,7,7,7,7,7]
        # lengths: [1, 3] 例如 [[5.2, 5.2, 8.1]]
        # angles: [1, 3] 例如 [[90, 90, 120]]
        
        frac_coords = outputs['frac_coords'].detach().cpu()
        num_atoms = outputs['num_atoms'].detach().cpu()
        atom_types = outputs['atom_types'].detach().cpu()
        lengths = outputs['lengths'].detach().cpu()
        angles = outputs['angles'].detach().cpu()
        
        # 5. 验证生成的结构符合输入组成
        assert num_atoms.item() == gt_num_atoms.item(), \
            f"样本{sample_id}: 生成的原子数{num_atoms.item()}不等于输入{gt_num_atoms.item()}"
        
        assert torch.equal(atom_types.cpu(), gt_atom_types.cpu()), \
            f"样本{sample_id}: 生成的原子类型不匹配输入组成"
        
        # 6. 收集结果
        all_frac_coords.append(frac_coords)  # [num_atoms, 3]
        all_num_atoms.append(num_atoms)      # [1]
        all_atom_types.append(atom_types)    # [num_atoms]
        all_lengths.append(lengths)          # [1, 3]
        all_angles.append(angles)            # [1, 3]
        
    
    # 7. 将结果组织成与原evaluate.py一致的格式
    # 注意：因为没有批处理，需要手动添加批次维度
    
    # 将列表转换为张量
    # all_num_atoms: list of [1] -> stack -> [num_samples, 1] -> transpose -> [1, num_samples]
    num_atoms_tensor = torch.stack(all_num_atoms, dim=0).squeeze(1).unsqueeze(0)  # [1, num_samples]
    
    # all_lengths: list of [1, 3] -> stack -> [num_samples, 1, 3] -> squeeze -> [num_samples, 3] -> unsqueeze -> [1, num_samples, 3]
    lengths_tensor = torch.cat(all_lengths, dim=0).unsqueeze(0)  # [1, num_samples, 3]
    
    # all_angles: list of [1, 3] -> similar process
    angles_tensor = torch.cat(all_angles, dim=0).unsqueeze(0)  # [1, num_samples, 3]
    
    # 对于不定长的frac_coords和atom_types，需要拼接
    frac_coords_tensor = torch.cat(all_frac_coords, dim=0).unsqueeze(0)  # [1, total_atoms, 3]
    atom_types_tensor = torch.cat(all_atom_types, dim=0).unsqueeze(0)    # [1, total_atoms]
    
    return (
        frac_coords_tensor,  # [1, total_atoms, 3]
        num_atoms_tensor,    # [1, num_samples]
        atom_types_tensor,   # [1, total_atoms]
        lengths_tensor,      # [1, num_samples, 3]
        angles_tensor,       # [1, num_samples, 3]
        None,  # all_frac_coords_stack (轨迹，不需要)
        None   # all_atom_types_stack (轨迹，不需要)
    )


def generate_submission(frac_coords, num_atoms, atom_types, lengths, angles, compositions_dict):
    """
    生成比赛提交文件 submission.csv
    
    Args:
        frac_coords: [1, total_atoms, 3] 所有原子的分数坐标
        num_atoms: [1, num_samples] 每个结构的原子数
        atom_types: [1, total_atoms] 所有原子的类型
        lengths: [1, num_samples, 3] 晶格参数
        angles: [1, num_samples, 3] 晶格角度
        compositions_dict: 组成信息字典，用于获取ID前缀
    """
    
    # 去除批次维度
    frac_coords = frac_coords.squeeze(0)  # [total_atoms, 3]
    num_atoms = num_atoms.squeeze(0)      # [num_samples]
    atom_types = atom_types.squeeze(0)    # [total_atoms]
    lengths = lengths.squeeze(0)          # [num_samples, 3]
    angles = angles.squeeze(0)            # [num_samples, 3]
    
    # 转换为CPU和numpy
    frac_coords = frac_coords.cpu().numpy()
    num_atoms = num_atoms.cpu().long()
    atom_types = atom_types.cpu().numpy().astype(int)
    lengths = lengths.cpu().numpy()
    angles = angles.cpu().numpy()
    
    # 验证原子总数
    total_atoms = num_atoms.sum().item()
    assert total_atoms == frac_coords.shape[0], "原子总数不匹配"
    assert len(atom_types) == total_atoms, "原子类型数量不匹配"
    
    # 获取ID前缀（A或B）
    prefix = next(iter(compositions_dict))[0]  # 'A' or 'B'
    
    # 生成Structure对象并转换为CIF
    rows = []
    current_idx = 0
    
    for i in tqdm(range(num_atoms.shape[0]), desc="生成CIF文件"):
        n_i = num_atoms[i].item()
        
        # 提取第i个晶体的数据
        coords = frac_coords[current_idx:current_idx + n_i]
        types = atom_types[current_idx:current_idx + n_i]
        
        # 创建晶格（使用lengths和angles）
        a, b, c = lengths[i]
        alpha, beta, gamma = angles[i]
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        
        # 转换为元素列表
        species = [Element.from_Z(z) for z in types]
        
        # 创建Structure
        structure = Structure(lattice, species, coords)
        
        # 生成CIF字符串
        cif = structure.to(fmt="cif")
        
        # 添加到结果
        ID = f"{prefix}-{i+1}"
        rows.append([ID, cif])
        
        current_idx += n_i
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows, columns=["ID", "cif"])
    df.to_csv("submission.csv", index=False)
    
    return len(rows)


def main():
    """主函数"""
    # 1. 加载组成数据
    composition_path = Path(DATA_PATH) / "composition.json"
    with open(composition_path, 'r') as f:
        compositions_dict = json.load(f)
    
    # 2. 加载模型
    model_path = Path(MODEL_PATH)
    model, _, _ = load_model(model_path, load_data=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 3. 生成晶体结构
    print("开始生成晶体结构...")
    start_time = time.time()
    
    results = generation_for_competition(model, compositions_dict)
    
    frac_coords, num_atoms, atom_types, lengths, angles, _, _ = results
    
    # 4. 保存.pt文件（与evaluate.py格式一致）
    elapsed_time = time.time() - start_time
    output_name = 'competition_gen.pt' if not DRY_RUN else 'competition_gen_dryrun.pt'
    output_path = Path('.') / output_name
    
    torch.save({
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'time': elapsed_time,
        'dry_run': DRY_RUN
    }, output_path)
    
    print(f"结果已保存到: {output_path}")
    
    # 5. 生成submission.csv
    num_structures = generate_submission(
        frac_coords, num_atoms, atom_types, lengths, angles, compositions_dict
    )
    
    print(f"成功生成 {num_structures} 个晶体结构")
    print(f"submission.csv 已生成")
    print(f"总用时: {elapsed_time:.2f} 秒")


if __name__ == '__main__':
    main()