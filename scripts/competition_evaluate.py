"""
Competition evaluation script for CDVAE - 批处理版本
核心改进：将逐个生成改为批量生成，提升速度

批处理架构说明：
1. 原版本：逐个样本处理，每个样本单独调用模型
2. 新版本：将多个样本打包成批次，一次性处理多个样本
3. 性能提升：GPU并行计算能力得到充分利用，速度提升数倍

关键技术点：
- 使用offsets记录每个样本在拼接数据中的位置
- 批量生成潜在向量z，维度为[batch_size, hidden_dim]
- 模型一次处理整个批次，返回拼接的结果
- 正确解包批次结果为单个样本
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from pymatgen.core import Composition, Element, Structure, Lattice
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent))
from eval_utils import load_model


# ========== 配置 ==========
# 数据路径：比赛数据集的位置，包含composition.json和pattern文件夹
DATA_PATH = "/home/ma-user/work/cdvae/docs/比赛用例/data/A"

# 模型路径：训练好的CDVAE模型checkpoint位置
MODEL_PATH = "/home/ma-user/work/cdvae/outputs/mp20epochs100/08-06-18-09-53/outputs/mp20epochs100/08-06-18-09-53"

# 调试模式：设为True时只处理前10个样本，用于快速测试
DRY_RUN = False  

# 批处理大小：一次处理的样本数量
# 注意事项：
# - GPU内存充足时可以增大（如256, 512）
# - GPU内存不足时需要减小（如32, 64）
# - 批处理大小直接影响生成速度
BATCH_SIZE = 128  

# Langevin dynamics参数配置
# 这些参数控制从潜在空间到晶体结构的生成过程
LD_KWARGS = SimpleNamespace(
    n_step_each=10,      # 每个噪声水平的步数，越大生成质量越高但速度越慢
    step_lr=1e-4,        # 步长，控制每步的更新幅度
    min_sigma=0,         # 最小噪声水平，通常设为0
    save_traj=False,     # 是否保存生成轨迹，设为False节省内存
    disable_bar=True     # 禁用进度条，避免与外层进度条冲突
)


def parse_composition(comp_str):
    """
    解析组成字符串为原子数量和类型
    
    输入示例: "Sr4 Be2 Re2 N8"
    输出示例: 
        num_atoms: 16  # 总原子数
        atom_types: [38,38,38,38,4,4,75,75,7,7,7,7,7,7,7,7]  # 原子序数列表
    
    Args:
        comp_str: 化学组成字符串，格式为"元素1数量1 元素2数量2 ..."
    
    Returns:
        tuple: (原子总数, 原子序数列表)
    
    工作原理：
    1. 使用pymatgen解析组成字符串
    2. 将每个元素转换为原子序数
    3. 根据数量展开成列表
    """
    comp = Composition(comp_str)
    atom_list = []
    
    for element, count in comp.items():
        # 获取元素的原子序数（如Sr=38, Be=4, Re=75, N=7）
        atomic_num = Element(element).Z
        # 根据数量重复原子序数
        atom_list.extend([atomic_num] * int(count))
    
    return len(atom_list), atom_list


def prepare_batch_data(sample_ids, compositions_dict):
    """
    准备一个批次的数据，将多个样本的组成信息打包
    
    批处理的核心挑战：
    不同样本的原子数不同，需要特殊处理变长序列
    
    解决方案：
    1. 将所有样本的原子类型拼接成一个长列表
    2. 记录每个样本在列表中的起始位置（offsets）
    3. 记录每个样本的原子数（num_atoms）
    
    示例：
    样本1: "H2O" -> 3个原子 [1,1,8]
    样本2: "NH3" -> 4个原子 [7,1,1,1]
    
    拼接结果：
    batch_atom_types = [1,1,8,7,1,1,1]
    batch_num_atoms = [3, 4]
    batch_offsets = [0, 3]  # 样本1从0开始，样本2从3开始
    
    Args:
        sample_ids: 批次中的样本ID列表
        compositions_dict: 包含所有样本组成信息的字典
    
    Returns:
        tuple: (
            batch_num_atoms: tensor，每个样本的原子数 [batch_size]
            batch_atom_types: tensor，所有原子类型拼接 [total_atoms_in_batch]
            batch_offsets: list，每个样本的起始位置
        )
    """
    batch_num_atoms = []
    batch_atom_types = []
    batch_offsets = [0]  # 第一个样本从位置0开始
    
    for sample_id in sample_ids:
        # 获取组成字符串（使用niggli reduced cell的组成，即第一个）
        comp_str = compositions_dict[sample_id]["composition"][0]
        
        # 解析组成
        num_atoms, atom_types = parse_composition(comp_str)
        
        # 收集数据
        batch_num_atoms.append(num_atoms)
        batch_atom_types.extend(atom_types)
        
        # 记录下一个样本的起始位置
        batch_offsets.append(len(batch_atom_types))
    
    return (
        torch.tensor(batch_num_atoms),
        torch.tensor(batch_atom_types),
        batch_offsets[:-1]  # 不需要最后一个offset（它等于总长度）
    )


def batch_generation(model, sample_ids, compositions_dict):
    """
    批量生成晶体结构 - 批处理的核心函数
    
    核心逻辑：
    1. 准备批次数据（多个样本的组成信息）
    2. 生成批量z向量（潜在表示）
    3. 调用模型批量生成晶体结构
    4. 返回批量结果
    
    性能优化说明：
    - 一次生成batch_size个z向量，而不是逐个生成
    - 模型的Langevin dynamics可以并行处理整个批次
    - GPU利用率大幅提升
    
    Args:
        model: CDVAE模型
        sample_ids: 当前批次的样本ID列表
        compositions_dict: 组成信息字典
    
    Returns:
        dict: 包含批次生成结果的字典
            - frac_coords: 分数坐标 [total_atoms_in_batch, 3]
            - num_atoms: 每个样本的原子数 [batch_size]
            - atom_types: 原子类型 [total_atoms_in_batch]
            - lengths: 晶格长度 [batch_size, 3]
            - angles: 晶格角度 [batch_size, 3]
            - offsets: 用于解包的偏移量列表
    """
    batch_size = len(sample_ids)
    
    # 步骤1：准备批次数据
    # 将多个样本的组成信息打包成批次格式
    batch_num_atoms, batch_atom_types, batch_offsets = prepare_batch_data(
        sample_ids, compositions_dict
    )
    
    # 将数据移动到GPU（如果可用）
    if torch.cuda.is_available():
        batch_num_atoms = batch_num_atoms.cuda()
        batch_atom_types = batch_atom_types.cuda()
    
    # 步骤2：生成批量潜在向量
    # z的形状: [batch_size, hidden_dim]
    # 例如: [128, 256] 表示128个样本，每个用256维向量表示
    z = torch.randn(batch_size, model.hparams.hidden_dim, device=model.device)
    
    # 步骤3：批量Langevin dynamics生成
    # 这是最耗时的步骤，但批处理大幅提升了效率
    # gt_num_atoms和gt_atom_types作为条件，确保生成的结构符合给定组成
    outputs = model.langevin_dynamics(
        z, 
        LD_KWARGS,
        gt_num_atoms=batch_num_atoms,    # 强制每个样本的原子数
        gt_atom_types=batch_atom_types   # 强制原子类型序列
    )
    
    # 步骤4：返回结果（转移到CPU便于后续处理）
    # detach()断开梯度连接，cpu()移动到CPU
    return {
        'frac_coords': outputs['frac_coords'].detach().cpu(),
        'num_atoms': outputs['num_atoms'].detach().cpu(), 
        'atom_types': outputs['atom_types'].detach().cpu(),
        'lengths': outputs['lengths'].detach().cpu(),
        'angles': outputs['angles'].detach().cpu(),
        'offsets': batch_offsets  # 保留offsets用于解包
    }


def unpack_batch_results(batch_outputs, sample_ids):
    """
    将批次结果解包为单个样本的列表
    
    关键挑战：
    批次输出中所有样本的原子数据是拼接在一起的，
    需要根据每个样本的原子数和偏移量正确分割。
    
    解包示例：
    假设批次有2个样本：
    - 样本1: 3个原子，offset=0
    - 样本2: 4个原子，offset=3
    
    batch_outputs['frac_coords']形状为[7, 3]（总共7个原子）
    需要分割为：
    - 样本1的坐标: [0:3, :] -> [3, 3]
    - 样本2的坐标: [3:7, :] -> [4, 3]
    
    Args:
        batch_outputs: batch_generation返回的批次输出
        sample_ids: 批次中的样本ID列表
    
    Returns:
        list: 解包后的单个样本结果列表，每个元素是一个字典
    """
    results = []
    batch_size = len(sample_ids)
    
    for i in range(batch_size):
        # 计算当前样本的原子数据范围
        # start_idx: 在拼接数组中的起始位置
        start_idx = batch_outputs['offsets'][i]
        
        # num_atoms: 当前样本的原子数
        num_atoms = batch_outputs['num_atoms'][i].item()
        
        # end_idx: 在拼接数组中的结束位置
        end_idx = start_idx + num_atoms
        
        # 提取当前样本的数据
        # 使用切片[start_idx:end_idx]从拼接的数据中提取
        sample_result = {
            'sample_id': sample_ids[i],
            'frac_coords': batch_outputs['frac_coords'][start_idx:end_idx],  # [num_atoms, 3]
            'num_atoms': num_atoms,
            'atom_types': batch_outputs['atom_types'][start_idx:end_idx],    # [num_atoms]
            'lengths': batch_outputs['lengths'][i],                          # [3]
            'angles': batch_outputs['angles'][i]                             # [3]
        }
        results.append(sample_result)
    
    return results


def generation_for_competition_batch(model, compositions_dict):
    """
    主生成函数 - 批处理版本
    
    改进点说明：
    1. 原版本：逐个处理2000个样本，每个样本调用一次模型
    2. 新版本：按批次处理，每批128个样本，只需调用约16次模型
    3. 速度提升：理论上可提升100倍以上（取决于GPU和批次大小）
    
    处理流程：
    1. 将所有样本ID分成多个批次
    2. 对每个批次调用batch_generation
    3. 解包批次结果
    4. 整理成统一格式
    
    Args:
        model: CDVAE模型
        compositions_dict: 所有样本的组成信息
    
    Returns:
        tuple: 与原evaluate.py格式一致的结果
    """
    # 获取所有样本ID
    sample_ids = list(compositions_dict.keys())
    
    # DRY_RUN模式：用于快速测试
    if DRY_RUN:
        sample_ids = sample_ids[:10]
        print(f"DRY_RUN模式：只处理前{len(sample_ids)}个样本")
    
    # 存储所有结果
    all_results = []
    
    # 按批次处理
    # range(0, len(sample_ids), BATCH_SIZE)生成批次起始索引
    # 例如：0, 128, 256, 384, ...
    for i in tqdm(range(0, len(sample_ids), BATCH_SIZE), desc="批量生成晶体"):
        # 获取当前批次的样本ID
        # 使用切片获取，最后一个批次可能不足BATCH_SIZE
        batch_sample_ids = sample_ids[i:i + BATCH_SIZE]
        
        # 批量生成
        # 这是核心步骤，一次处理整个批次
        batch_outputs = batch_generation(model, batch_sample_ids, compositions_dict)
        
        # 解包批次结果
        # 将批次输出分割为单个样本
        batch_results = unpack_batch_results(batch_outputs, batch_sample_ids)
        
        # 收集结果
        all_results.extend(batch_results)
    
    # 整理成原evaluate.py的格式
    # 添加批次维度[1, ...]以保持兼容性
    return organize_results(all_results)


def organize_results(all_results):
    """
    将结果整理成与原evaluate.py一致的格式
    
    格式要求：
    - 所有张量需要有批次维度 [1, ...]
    - frac_coords和atom_types是拼接的
    - num_atoms, lengths, angles是堆叠的
    
    数据组织示例：
    输入：list of dicts，每个dict包含一个样本的数据
    输出：tuple of tensors，格式与原evaluate.py一致
    
    Args:
        all_results: 所有样本的结果列表
    
    Returns:
        tuple: (frac_coords, num_atoms, atom_types, lengths, angles, None, None)
            - frac_coords: [1, total_atoms, 3]
            - num_atoms: [1, num_samples]
            - atom_types: [1, total_atoms]
            - lengths: [1, num_samples, 3]
            - angles: [1, num_samples, 3]
            - 最后两个None是轨迹数据，不需要
    """
    # 收集所有数据
    all_frac_coords = []  # 将存储所有样本的分数坐标
    all_num_atoms = []     # 将存储所有样本的原子数
    all_atom_types = []    # 将存储所有样本的原子类型
    all_lengths = []       # 将存储所有样本的晶格长度
    all_angles = []        # 将存储所有样本的晶格角度
    
    # 遍历每个样本的结果
    for result in all_results:
        all_frac_coords.append(result['frac_coords'])
        all_num_atoms.append(result['num_atoms'])
        all_atom_types.append(result['atom_types'])
        all_lengths.append(result['lengths'])
        all_angles.append(result['angles'])
    
    # 拼接并添加批次维度
    # cat: 拼接不定长的数据（frac_coords, atom_types）
    # stack/tensor: 堆叠定长的数据（lengths, angles, num_atoms）
    # unsqueeze(0): 添加批次维度
    frac_coords = torch.cat(all_frac_coords, dim=0).unsqueeze(0)    # [1, total_atoms, 3]
    num_atoms = torch.tensor(all_num_atoms).unsqueeze(0)            # [1, num_samples]
    atom_types = torch.cat(all_atom_types, dim=0).unsqueeze(0)      # [1, total_atoms]
    lengths = torch.stack(all_lengths, dim=0).unsqueeze(0)          # [1, num_samples, 3]
    angles = torch.stack(all_angles, dim=0).unsqueeze(0)            # [1, num_samples, 3]
    
    # 返回格式与原evaluate.py一致
    return frac_coords, num_atoms, atom_types, lengths, angles, None, None


def generate_submission(frac_coords, num_atoms, atom_types, lengths, angles, compositions_dict):
    """
    生成比赛提交文件 submission.csv
    
    文件格式要求：
    - CSV文件，包含两列：ID和cif
    - ID格式："{前缀}-{编号}"，如"A-1", "A-2", ...
    - cif列：晶体结构的CIF格式字符串
    
    处理流程：
    1. 去除批次维度
    2. 转换为numpy数组
    3. 逐个创建Structure对象
    4. 转换为CIF格式
    5. 保存为CSV
    
    Args:
        frac_coords: [1, total_atoms, 3] 所有原子的分数坐标
        num_atoms: [1, num_samples] 每个结构的原子数
        atom_types: [1, total_atoms] 所有原子的类型
        lengths: [1, num_samples, 3] 晶格参数a,b,c
        angles: [1, num_samples, 3] 晶格角度α,β,γ
        compositions_dict: 组成信息字典，用于获取ID前缀
    
    Returns:
        int: 生成的结构数量
    """
    # 步骤1：去除批次维度
    # squeeze(0)移除第一个维度（批次维度）
    frac_coords = frac_coords.squeeze(0)  # [total_atoms, 3]
    num_atoms = num_atoms.squeeze(0)      # [num_samples]
    atom_types = atom_types.squeeze(0)    # [total_atoms]
    lengths = lengths.squeeze(0)          # [num_samples, 3]
    angles = angles.squeeze(0)            # [num_samples, 3]
    
    # 步骤2：转换为numpy数组
    # pymatgen需要numpy数组而不是torch张量
    frac_coords = frac_coords.numpy()
    num_atoms = num_atoms.long()  # 确保是整数类型
    atom_types = atom_types.numpy().astype(int)
    lengths = lengths.numpy()
    angles = angles.numpy()
    
    # 步骤3：获取ID前缀
    # 从第一个样本ID中提取前缀（'A'或'B'）
    prefix = next(iter(compositions_dict))[0]  # 'A' or 'B'
    
    # 步骤4：生成CIF文件
    rows = []  # 存储CSV的行
    current_idx = 0  # 当前在拼接数组中的位置
    
    # 遍历每个样本
    for i in tqdm(range(num_atoms.shape[0]), desc="生成CIF文件"):
        # 获取当前样本的原子数
        n_i = num_atoms[i].item()
        
        # 提取当前样本的数据
        # 使用切片从拼接的数组中提取
        coords = frac_coords[current_idx:current_idx + n_i]  # [n_i, 3]
        types = atom_types[current_idx:current_idx + n_i]    # [n_i]
        
        # 创建晶格对象
        # Lattice.from_parameters使用晶格参数创建
        a, b, c = lengths[i]
        alpha, beta, gamma = angles[i]
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        
        # 转换原子序数为元素对象
        species = [Element.from_Z(z) for z in types]
        
        # 创建Structure对象
        # Structure包含晶格、元素和坐标信息
        structure = Structure(lattice, species, coords)
        
        # 生成CIF字符串
        ID = f"{prefix}-{i+1}"  # 样本ID，如"A-1"
        cif = structure.to(fmt="cif")  # 转换为CIF格式
        rows.append([ID, cif])
        
        # 更新位置索引
        current_idx += n_i
    
    # 步骤5：保存为CSV文件
    df = pd.DataFrame(rows, columns=["ID", "cif"])
    df.to_csv("submission.csv", index=False)
    
    print(f"submission.csv 已生成，包含 {len(rows)} 个结构")
    return len(rows)


def main():
    """
    主函数 - 程序入口
    
    执行流程：
    1. 加载组成数据（composition.json）
    2. 加载CDVAE模型
    3. 批量生成晶体结构
    4. 保存生成结果（.pt文件）
    5. 生成提交文件（submission.csv）
    
    性能说明：
    - 批处理版本相比原版本速度提升显著
    - 2000个样本预计耗时：原版~1小时，批处理版~5分钟（取决于GPU）
    """
    # 步骤1：加载组成数据
    composition_path = Path(DATA_PATH) / "composition.json"
    with open(composition_path, 'r') as f:
        compositions_dict = json.load(f)
    print(f"加载了 {len(compositions_dict)} 个样本的组成信息")
    
    # 步骤2：加载模型
    model_path = Path(MODEL_PATH)
    print(f"正在加载模型: {model_path}")
    model, _, _ = load_model(model_path, load_data=False)
    
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU进行生成")
    else:
        print("警告：使用CPU进行生成，速度会很慢")
    
    # 步骤3：批量生成晶体结构
    print("开始批量生成晶体结构...")
    print(f"批处理大小: {BATCH_SIZE}")
    start_time = time.time()
    
    # 调用批处理生成函数
    results = generation_for_competition_batch(model, compositions_dict)
    
    # 解包结果
    frac_coords, num_atoms, atom_types, lengths, angles, _, _ = results
    
    # 步骤4：保存结果到.pt文件
    elapsed_time = time.time() - start_time
    output_name = 'competition_gen.pt' if not DRY_RUN else 'competition_gen_dryrun.pt'
    
    # 保存所有生成的数据
    torch.save({
        'frac_coords': frac_coords,     # 分数坐标
        'num_atoms': num_atoms,         # 原子数
        'atom_types': atom_types,       # 原子类型
        'lengths': lengths,             # 晶格长度
        'angles': angles,               # 晶格角度
        'time': elapsed_time,           # 生成用时
        'dry_run': DRY_RUN,            # 是否为测试模式
        'batch_size': BATCH_SIZE       # 批处理大小
    }, output_name)
    
    print(f"结果已保存到: {output_name}")
    
    # 步骤5：生成submission.csv
    print("正在生成submission.csv...")
    num_structures = generate_submission(
        frac_coords, num_atoms, atom_types, lengths, angles, compositions_dict
    )
    
    # 打印统计信息
    print("="*50)
    print(f"成功生成 {num_structures} 个晶体结构")
    print(f"总用时: {elapsed_time:.2f} 秒")
    print(f"平均每个结构: {elapsed_time/num_structures:.3f} 秒")
    print(f"批处理大小: {BATCH_SIZE}")
    print("="*50)


if __name__ == '__main__':
    main()