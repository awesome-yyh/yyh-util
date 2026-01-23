import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns

class RoPEVisualizer:
    def __init__(self, dim: int = 64, max_len: int = 1000, base: int = 10000):
        """RoPE可视化器
        
        Args:
            dim: 编码维度(必须是偶数)
            max_len: 最大序列长度
            base: RoPE基值(rope_theta)
        """
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # 预计算频率
        self.freqs = self._get_freqs()
        
    def _get_freqs(self) -> np.ndarray:
        """计算RoPE频率"""
        return 1.0 / (self.base ** (np.arange(0, self.dim, 2) / self.dim))
    
    def get_rotation_matrix(self, pos: int) -> np.ndarray:
        """获取位置pos的旋转矩阵"""
        theta = pos * self.freqs
        cos = np.cos(theta)
        sin = np.sin(theta)
        
        rot_matrices = np.zeros((self.dim//2, 2, 2))
        rot_matrices[:, 0, 0] = cos
        rot_matrices[:, 0, 1] = -sin
        rot_matrices[:, 1, 0] = sin
        rot_matrices[:, 1, 1] = cos
        
        return rot_matrices
    
    def apply_rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """应用RoPE变换"""
        x_reshaped = x.reshape(-1, 2)
        rot_matrices = self.get_rotation_matrix(pos)
        rotated = np.einsum('ijk,ik->ij', rot_matrices, x_reshaped)
        return rotated.flatten()
    
    def compute_attention_score(self, q_pos: int, k_pos: int) -> float:
        """计算两个位置间的注意力分数"""
        # 生成随机query和key向量
        query = np.random.randn(self.dim)
        key = np.random.randn(self.dim)
        
        # 应用RoPE变换
        q_rotated = self.apply_rope(query, q_pos)
        k_rotated = self.apply_rope(key, k_pos)
        
        # 计算注意力分数
        return np.dot(q_rotated, k_rotated)
    
    def visualize_relative_position(self):
        """可视化相对位置编码特性"""
        positions = np.arange(10)
        attention_matrix = np.zeros((len(positions), len(positions)))
        
        for i in positions:
            for j in positions:
                attention_matrix[i, j] = self.compute_attention_score(i, j)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, annot=True, fmt='.2f')
        plt.title('Attention Scores Matrix')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()
        
        # 验证R(n-m) = Rm^T * Rn
        pos1, pos2 = 3, 7
        R_m = self.get_rotation_matrix(pos1)
        R_n = self.get_rotation_matrix(pos2)
        R_diff_direct = self.get_rotation_matrix(pos2 - pos1)
        R_diff_composed = np.einsum('ijk,ikl->ijl', R_m.transpose(0, 2, 1), R_n)
        
        print(f"\nVerifying R({pos2}-{pos1}) = R({pos1})^T * R({pos2}):")
        print(f"Max difference: {np.abs(R_diff_direct - R_diff_composed).max():.8f}")
    
    def visualize_extrapolation(self, test_lengths=[100, 1000, 10000]):
        """可视化长度外推性"""
        # 选择不同维度观察周期性
        dims_to_show = [0, self.dim//4, self.dim//2-1]  # 高频、中频、低频维度
        
        plt.figure(figsize=(15, 10))
        for dim_idx in dims_to_show:
            positions = np.arange(max(test_lengths))
            theta = positions * self.freqs[dim_idx]
            values = np.cos(theta)
            
            plt.plot(positions, values, 
                    label=f'Dimension {dim_idx*2} (freq={self.freqs[dim_idx]:.1e})')
        
        plt.title('RoPE Periodic Patterns')
        plt.xlabel('Position')
        plt.ylabel('Cosine Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def visualize_decay(self, query_pos: int = 0, max_distance: int = 100):
        """可视化远程衰减性"""
        distances = np.arange(max_distance)
        attention_scores = []
        
        # 固定query位置，计算不同距离的attention scores
        for dist in distances:
            score = self.compute_attention_score(query_pos, query_pos + dist)
            attention_scores.append(score)
        
        plt.figure(figsize=(12, 6))
        plt.plot(distances, attention_scores)
        plt.title(f'Attention Score Decay (from position {query_pos})')
        plt.xlabel('Relative Distance')
        plt.ylabel('Attention Score')
        plt.grid(True)
        plt.show()

    def compare_different_bases(self, test_lengths=[1000, 10000, 100000], 
                            base_values=[10000, 100000, 1000000]):
        """比较不同base值对RoPE周期性的影响
        
        Args:
            test_lengths: 要测试的序列长度列表
            base_values: 要比较的base值列表
        """
        # 选择不同维度观察周期性
        dims_to_show = [0, self.dim//4, self.dim//2-1]  # 高频、中频、低频维度
        
        # 创建子图
        fig, axes = plt.subplots(len(base_values), 1, figsize=(15, 5*len(base_values)))
        fig.suptitle('Comparison of Different RoPE Base Values', 
                    fontsize=16, y=0.95)
        
        # 为每个base值创建一个子图
        for idx, base in enumerate(base_values):
            # 更新频率
            self.base = base
            self.freqs = self._get_freqs()
            
            # 在对应的子图上绘制
            ax = axes[idx]
            
            for dim_idx in dims_to_show:
                positions = np.arange(max(test_lengths))
                theta = positions * self.freqs[dim_idx]
                values = np.cos(theta)
                
                ax.plot(positions, values, 
                    label=f'Dimension {dim_idx*2} (freq={self.freqs[dim_idx]:.1e})')
            
            ax.set_title(f'Base Value = {base:,}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Cosine Value')
            ax.legend()
            ax.grid(True)
            
            # 添加垂直线标记不同的序列长度
            for length in test_lengths:
                ax.axvline(x=length, color='r', linestyle='--', alpha=0.3)
                ax.text(length, ax.get_ylim()[1], f'len={length}', 
                    rotation=90, va='top', ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # 打印频率信息
        print("\nFrequency comparison for different dimensions:")
        print("\nDim | " + " | ".join(f"Base={base:,}".ljust(15) for base in base_values))
        print("-" * (5 + 17 * len(base_values)))
        
        for dim_idx in dims_to_show:
            freqs = []
            for base in base_values:
                freq = 1.0 / (base ** (2*dim_idx/self.dim))
                freqs.append(f"{freq:.2e}")
            print(f"{dim_idx*2:3d} | " + " | ".join(f"{freq}".ljust(15) for freq in freqs))


def main():
    # 创建RoPE可视化器
    rope_viz = RoPEVisualizer(dim=64, max_len=1000, base=10000)
    
    print("1. 演示相对位置编码特性")
    print("""对角线模式：对角线上的值（自注意）应该最高，因为位置差为0
对称性：位置i到j的注意力分数应与位置j到i的分数呈一定对称关系
距离衰减：随着离对角线越远（位置差越大），注意力分数应该逐渐减小
周期性：可能会观察到一定的周期性模式，这是由RoPE的三角函数特性导致的""")
    rope_viz.visualize_relative_position()
    
    print("\n2. 演示长度外推性")
    print("""不同频率：
低维度（如dim=0）：高频振荡，周期短，适合捕捉局部位置关系
高维度（如dim=dim//2-1）：低频振荡，周期长，适合捕捉全局位置关系
中间维度：中等频率，在局部和全局之间平衡
外推能力：波形能够延续到训练长度之外，说明RoPE能处理更长序列""")
    rope_viz.visualize_extrapolation()
    
    print("\n3. 演示远程衰减性")
    print("""初始衰减：随着距离增加，注意力分数迅速下降
震荡特性：可能会出现周期性的小幅震荡
长尾效应：在较远距离处，注意力分数趋于稳定但保持较低水平
有界性：注意力分数应该在一定范围内波动，不会发散""")
    rope_viz.visualize_decay()
    
    # 比较不同base值的影响
    test_lengths = [1000, 10000, 100000]
    base_values = [10000, 100000, 1000000]
    print("\n比较不同base值的影响")
    print("""base值增大的效果：
周期延长：更大的base值会使波形周期变长
衰减减缓：位置编码的衰减速度变慢
分辨率权衡：更长的周期意味着相对位置的分辨率可能降低""")
    print("""对每个base值的子图，可以对比：

波形周期：
base=10000：基准周期
base=100000：周期约变为10倍
base=1000000：周期约变为100倍
不同维度的行为：
低维：周期变化最明显
高维：周期变化相对较小
中间维度：呈现中等程度的变化""")
    rope_viz.compare_different_bases(test_lengths, base_values)

    # # 额外：展示不同base值对外推性的影响
    # print("\n4. 比较不同base值的影响")
    # for base in [10000, 100000, 1000000]:
    #     rope_viz = RoPEVisualizer(dim=64, max_len=1000, base=base)
    #     print(f"\nUsing base={base}")
    #     rope_viz.visualize_extrapolation([1000, 10000, 100000])

if __name__ == "__main__":
    main()