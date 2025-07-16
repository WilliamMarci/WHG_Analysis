"""
# 使用示例:
# 1. 确保安装了必要的包:
#    pip install uproot awkward coffea matplotlib numpy
#
# 2. 将LHE ROOT文件放在正确位置
#
# 3. 运行脚本:
#    python lhe_photon_analysis.py
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, DelphesSchema

def analyze_photon_pt_from_lhe(file_path, tree_name="LHEF"):
    """
    从LHE文件中分析光子PT分布
    
    参数:
    file_path: LHE ROOT文件路径
    tree_name: 树的名称，默认为"LHEF"
    
    返回:
    photon_pts: 光子PT数组
    """
    
    print(f"正在读取LHE文件: {file_path}")
    
    # 方法1: 使用coffea读取LHE文件
    try:
        events = NanoEventsFactory.from_root(
            file_path,
            treepath=tree_name,
        ).events()
        
        print(f"成功读取 {len(events)} 个事件")
        
        # 提取粒子信息
        # 在LHE格式中，通常包含以下字段：
        # - Particle_PID: 粒子ID
        # - Particle_Status: 粒子状态
        # - Particle_Px, Particle_Py, Particle_Pz: 动量分量
        # - Particle_E: 能量
        
        # 选择出射光子 (PID=22, Status=1)
        photon_mask = (events.Particle_PID == 22) & (events.Particle_Status == 1)
        
        # 计算横向动量
        px = ak.flatten(events.Particle_Px[photon_mask])
        py = ak.flatten(events.Particle_Py[photon_mask])
        pt = np.sqrt(px**2 + py**2)
        
        return ak.to_numpy(pt)
        
    except Exception as e:
        print(f"使用coffea读取失败: {e}")
        
        # 方法2: 使用uproot直接读取
        try:
            with uproot.open(file_path) as file:
                tree = file[tree_name]
                
                # 读取粒子数据
                pid = tree["Particle.PID"].array()
                status = tree["Particle.Status"].array()
                px = tree["Particle.Px"].array()
                py = tree["Particle.Py"].array()
                
                print(f"使用uproot成功读取 {len(pid)} 个事件")
                
                # 选择出射光子
                photon_mask = (pid == 22) & (status == 1)
                
                # 计算PT
                photon_px = ak.flatten(px[photon_mask])
                photon_py = ak.flatten(py[photon_mask])
                pt = np.sqrt(photon_px**2 + photon_py**2)
                
                return ak.to_numpy(pt)
                
        except Exception as e2:
            print(f"使用uproot读取也失败: {e2}")
            return None
def plot_photon_pt_distribution(photon_pts, output_file=None, pt_range=None):
    """
    Plot photon PT distribution

    Args:
    photon_pts: Array of photon PT values
    output_file: Output file name (optional)
    pt_range: Tuple (min_pt, max_pt) to set PT range for histograms (optional)
    """

    if photon_pts is None or len(photon_pts) == 0:
        print("No photon data found")
        return

    print(f"\nPhoton PT distribution statistics:")
    print(f"Total photons: {len(photon_pts)}")
    print(f"PT range: {photon_pts.min():.2f} - {photon_pts.max():.2f} GeV")
    print(f"Mean PT: {photon_pts.mean():.2f} GeV")
    print(f"Median PT: {np.median(photon_pts):.2f} GeV")
    print(f"PT std: {photon_pts.std():.2f} GeV")

    # Apply PT range if specified
    if pt_range is not None:
        min_pt, max_pt = pt_range
        photon_pts = photon_pts[(photon_pts >= min_pt) & (photon_pts <= max_pt)]

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # 1. Linear scale histogram
    ax1.hist(photon_pts, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True, range=pt_range)
    ax1.set_xlabel('Photon $p_T$ (GeV)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Photon Transverse Momentum Distribution (Linear Scale)')
    ax1.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f'Total: {len(photon_pts)}\nMean: {photon_pts.mean():.1f} GeV\nMedian: {np.median(photon_pts):.1f} GeV'
    ax1.text(0.65, 0.8, stats_text, transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 2. Log scale histogram
    counts, bins, _ = ax2.hist(photon_pts, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', range=pt_range)
    ax2.set_xlabel('Photon $p_T$ (GeV)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Photon Transverse Momentum Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    sorted_pts = np.sort(photon_pts)
    cumulative = np.arange(1, len(sorted_pts) + 1) / len(sorted_pts)
    ax3.plot(sorted_pts, cumulative, linewidth=2, color='green')
    ax3.set_xlabel('Photon $p_T$ (GeV)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Photon PT Cumulative Distribution Function')
    ax3.grid(True, alpha=0.3)

    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        pt_value = np.percentile(photon_pts, p)
        ax3.axvline(pt_value, color='red', linestyle='--', alpha=0.7)
        ax3.text(pt_value, p/100, f'{p}%\n{pt_value:.1f}', ha='center', va='bottom', fontsize=8)

    # 4. PT vs event index (trend check)
    ax4.scatter(range(min(1000, len(photon_pts))), photon_pts[:1000], alpha=0.5, s=1)
    ax4.set_xlabel('Photon Index (first 1000)')
    ax4.set_ylabel('Photon $p_T$ (GeV)')
    ax4.set_title('Photon PT vs Event Index')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
    plt.show()

def main():
    """
    主函数
    """
    # 文件路径 - 请根据实际情况修改
    file_path = "samples/whg_lo_pp_vlh_decayed.root"
    
    print("LHE文件光子PT分布分析程序")
    print("=" * 50)
    
    # 分析光子PT分布
    photon_pts = analyze_photon_pt_from_lhe(file_path)
    
    if photon_pts is not None:
        # 绘制分布图
        plot_photon_pt_distribution(photon_pts, "figures/photon_pt_distribution.png", pt_range=(0, 20))
        
        # 保存数据
        # np.savetxt("photon_pt_data.txt", photon_pts, header="Photon_PT_GeV")
        # print("光子PT数据已保存到: photon_pt_data.txt")
        
    else:
        print("无法读取LHE文件，请检查文件路径和格式")

# 如果直接运行此脚本
if __name__ == "__main__":
    main()

