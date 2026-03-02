import torch
import torch.nn as nn
import torch.nn.functional as F



try:
    from pytorch_wavelets import DWT1D
except ImportError:
    class DWT1D(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return x, [x]

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1, bias=False):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x): return self.pointwise(self.depthwise(x))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class iSQRTCovariance(nn.Module):
    def __init__(self, iter_num=3):
        super(iSQRTCovariance, self).__init__()
        self.iter_num = iter_num

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.size()

        # 1. 中心化
        x_mean = x.mean(dim=2, keepdim=True)
        x_centered = x - x_mean

        # 2. 计算无偏协方差矩阵 (B, C, C)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (L - 1 + 1e-5)

        # 3. iSQRT 预处理: 迹归一化 (Trace Normalization)
        # 为了保证 Newton-Schulz 迭代收敛，矩阵的谱半径必须 < 1
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        cov_norm = cov / (tr + 1e-5)  # 加上 epsilon 防止除零

        # 4. Newton-Schulz 迭代
        # Y_0 = A, Z_0 = I
        # Y_{k+1} = 0.5 * Y_k * (3I - Z_k * Y_k)
        # Z_{k+1} = 0.5 * (3I - Z_k * Y_k) * Z_k
        # 最终 Y 收敛于 A^{1/2}

        Y = cov_norm
        I = torch.eye(C, device=cov.device).unsqueeze(0).expand(B, -1, -1)
        Z = I

        for _ in range(self.iter_num):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # 5. 补偿迹的幅度 (反归一化)
        # sqrt(A) = sqrt(tr * A_norm) = sqrt(tr) * sqrt(A_norm)
        Y = Y * torch.sqrt(tr + 1e-5)

        # 6. 展平上三角或全矩阵
        return Y.view(B, -1)
class MultiScaleBlock_DSC_Lite(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(MultiScaleBlock_DSC_Lite, self).__init__()
        mid_channels = max(4, out_channels // 2)

        # Branch 1: Detail (小卷积核)
        self.branch1 = LiteConvBlock(
            in_channels, mid_channels,
            kernel_size=3, padding=1, stride=1
        )

        # Branch 2: Pattern (空洞卷积)
        self.branch2 = LiteConvBlock(
            in_channels, mid_channels,
            kernel_size=5, padding=4, dilation=2, stride=1
        )

        # Branch 3: MaxPool
        self.branch3 = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU6(inplace=True)
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv1d(mid_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True)
        )

        self.se = SELayer(out_channels, reduction)

    def forward(self, x):
        # 逻辑保持不变
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = self.fusion(torch.cat([x1, x2, x3], dim=1))
        return self.se(out)

class MultiBranch_Time_Net(nn.Module):
    def __init__(self, num_imu, num_strain, imu_nf, strain_nf, shared_nf, dropout):
        super(MultiBranch_Time_Net, self).__init__()
        self.imu_b1 = MultiScaleBlock_DSC_Lite(num_imu, imu_nf)
        self.imu_b2 = MultiScaleBlock_DSC_Lite(imu_nf, imu_nf * 2)

        strain_start = max(4, strain_nf // 2)
        self.strain_b1 = MultiScaleBlock_DSC_Lite(num_strain, strain_start)
        self.strain_b2 = MultiScaleBlock_DSC_Lite(strain_start, strain_nf * 2)

        in_shared = (imu_nf * 2) + (strain_nf * 2)
        self.shared = DepthwiseSeparableConv1d(in_shared, shared_nf, 3, 1, bias=False)
        self.bn = nn.BatchNorm1d(shared_nf)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_i = x[:, :, :9].transpose(1, 2)
        x_s = x[:, :, 9:].transpose(1, 2)
        x_i = self.imu_b2(self.imu_b1(x_i))
        x_s = self.strain_b2(self.strain_b1(x_s))
        return self.drop(self.act(self.bn(self.shared(torch.cat([x_i, x_s], dim=1)))))


# --- 独立验证用 Wrapper ---
class TimeDomainNet(nn.Module):
    """
    单独验证时域分支
    结构: Multi-Scale Extractor -> Global Avg Pool -> Classifier
    """
    def __init__(self, num_classes=8, num_imu=9, num_strain=2, base_nf=16, shared_nf=64, dropout=0.1):
        super(TimeDomainNet, self).__init__()
        # 1. 特征提取
        self.feature_extractor = MultiBranch_Time_Net(
            num_imu, num_strain, base_nf, base_nf // 2, shared_nf, dropout
        )
        # 2. 池化 (将时间维度压缩为1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shared_nf, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)  # (B, C, T)
        feat = self.pool(feat)  # (B, C, 1)
        return self.classifier(feat)
class LiteConvBlock(nn.Module):
    """
    升级版: 支持 dilation 参数
    结构: Depthwise (含 BN+ReLU6) -> Pointwise (含 BN+ReLU6)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(LiteConvBlock, self).__init__()
        self.net = nn.Sequential(
            # 1. Depthwise: 空间特征提取 (注意这里传入了 dilation)
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU6(inplace=True),

            # 2. Pointwise: 通道融合
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class EfficientCrossGating(nn.Module):
    def __init__(self, dim):
        super(EfficientCrossGating, self).__init__()
        # 降维比例 r=4
        reduced_dim = max(4, dim // 4)

        # 共享的上下文提取器
        self.context_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1) 全局描述符
            nn.Conv1d(dim, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv1d(reduced_dim, dim, 1),
            nn.Sigmoid()  # 生成 0-1 门控权重
        )

        self.out_proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x_main, x_context):
        # x_main: 主特征流
        # x_context: 辅助特征流 (提供上下文)

        # 1. 提取 Context 的通道注意力权重
        gate = self.context_net(x_context)  # (B, C, 1)

        # 2. 门控调制: 用 Context 的重要性去加权 Main
        out = x_main * gate

        # 3. 残差连接 + 投影
        return self.bn(self.out_proj(out) + x_main)

class GroupedCovariancePooling(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(GroupedCovariancePooling, self).__init__()
        assert in_channels % num_groups == 0, "Channels must be divisible by groups"
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups

    def forward(self, x):
        B, C, L = x.size()
        x = x.view(B, self.num_groups, self.group_channels, L)
        mean = x.mean(dim=3, keepdim=True)
        x_centered = x - mean  # (B, G, C_g, L)
        cov = torch.einsum('bgcl,bgdl->bgcd', x_centered, x_centered) / (L - 1 + 1e-5)
        return cov.view(B, -1)


class LiteWaveletCrossNet(nn.Module):
    def __init__(self, num_imu, num_strain, base_nf, dropout, wave_imu='haar', wave_strain='dmey'):
        super(LiteWaveletCrossNet, self).__init__()

        self.dwt_imu = DWT1D(wave=wave_imu, J=1, mode='symmetric')
        self.dwt_strain = DWT1D(wave=wave_strain, J=1, mode='symmetric')

        self.imu_cA = LiteConvBlock(num_imu, base_nf * 2)
        self.imu_cD = LiteConvBlock(num_imu, base_nf * 2)
        self.str_cA = LiteConvBlock(num_strain, base_nf)
        self.str_cD = LiteConvBlock(num_strain, base_nf)

        self.match_s_low = nn.Conv1d(base_nf, base_nf * 2, 1)
        self.match_s_high = nn.Conv1d(base_nf, base_nf * 2, 1)

        self.fuse_low = EfficientCrossGating(base_nf * 2)
        self.fuse_high = EfficientCrossGating(base_nf * 2)

        self.output_dim = base_nf * 4

    def forward(self, x):
        xi = x[..., :9].transpose(1, 2)
        xs = x[..., 9:].transpose(1, 2)

        i_cA, i_cD = self.dwt_imu(xi)[0], self.dwt_imu(xi)[1][0]
        s_cA, s_cD = self.dwt_strain(xs)[0], self.dwt_strain(xs)[1][0]

        min_L = min(i_cA.shape[2], s_cA.shape[2])
        i_cA, i_cD = i_cA[..., :min_L], i_cD[..., :min_L]
        s_cA, s_cD = s_cA[..., :min_L], s_cD[..., :min_L]

        i_feat_low = self.imu_cA(i_cA)
        i_feat_high = self.imu_cD(i_cD)
        s_feat_low = self.str_cA(s_cA)
        s_feat_high = self.str_cD(s_cD)

        s_feat_low = self.match_s_low(s_feat_low)
        s_feat_high = self.match_s_high(s_feat_high)

        out_low = self.fuse_low(i_feat_low, s_feat_low)
        out_high = self.fuse_high(i_feat_high, s_feat_high)

        return torch.cat([out_low, out_high], dim=1)


class FreqDomainNet_Lite(nn.Module):
    def __init__(self, num_classes=8, num_imu=9, num_strain=2, base_nf=16,
                 bottleneck_dim=32, groups=4, dropout=0.1,
                 wave_imu='haar', wave_strain='dmey'):
        super(FreqDomainNet_Lite, self).__init__()

        self.feature_extractor = LiteWaveletCrossNet(
            num_imu, num_strain, base_nf, dropout,
            wave_imu=wave_imu, wave_strain=wave_strain
        )

        feat_out_dim = self.feature_extractor.output_dim

        self.bottleneck = nn.Sequential(
            nn.Conv1d(feat_out_dim, bottleneck_dim, 1, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

        self.group_cov = GroupedCovariancePooling(bottleneck_dim, num_groups=groups)
        self.gap = nn.AdaptiveAvgPool1d(1)

        cov_dim = groups * (bottleneck_dim // groups) ** 2
        mean_dim = bottleneck_dim
        total_in = cov_dim + mean_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_in, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.bottleneck(feat)
        cov_feat = self.group_cov(feat)
        mean_feat = self.gap(feat).squeeze(-1)
        combined = torch.cat([cov_feat, mean_feat], dim=1)
        return self.classifier(combined)

class DualStreamFusionNet(nn.Module):
    def __init__(self,
                 num_classes=12,
                 num_imu=3,
                 num_strain=2,
                 # 时域参数
                 time_base_nf=16,
                 time_shared_nf=64,
                 # 频域参数
                 freq_base_nf=8,
                 freq_bottleneck_dim=32,
                 freq_groups=4,
                 # 全局参数
                 dropout=0.1):
        super(DualStreamFusionNet, self).__init__()

        # =======================================================
        # 1. 时域分支 (Time Stream)
        # =======================================================
        self.time_extractor = MultiBranch_Time_Net(
            num_imu, num_strain,
            imu_nf=time_base_nf,
            strain_nf=time_base_nf // 2,
            shared_nf=time_shared_nf,
            dropout=dropout
        )
        # 时域分支输出是 (B, C, L)，需要池化成向量 (B, C)
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.time_feat_dim = time_shared_nf

        # =======================================================
        # 2. 频域分支 (Frequency Stream - Lite Version)
        # =======================================================
        # 2.1 特征提取
        self.freq_extractor = LiteWaveletCrossNet(num_imu, num_strain, freq_base_nf, dropout)

        # 获取 LiteWaveletCrossNet 的输出维度 (base_nf * 4)
        freq_out_dim = self.freq_extractor.output_dim

        # 2.2 瓶颈层 (Bottleneck)
        self.freq_bottleneck = nn.Sequential(
            nn.Conv1d(freq_out_dim, freq_bottleneck_dim, 1, bias=False),
            nn.BatchNorm1d(freq_bottleneck_dim),
            nn.ReLU()
        )

        # 2.3 混合统计池化
        self.freq_group_cov = GroupedCovariancePooling(freq_bottleneck_dim, num_groups=freq_groups)
        self.freq_gap = nn.AdaptiveAvgPool1d(1)

        # 计算频域特征维度
        # Cov维度 + Mean维度
        self.freq_feat_dim = (freq_groups * (freq_bottleneck_dim // freq_groups) ** 2) + freq_bottleneck_dim

        # =======================================================
        # 3. 融合层 (Fusion & Classifier)
        # =======================================================
        total_dim = self.time_feat_dim + self.freq_feat_dim

        print(f"[Fusion Info] Time Dim: {self.time_feat_dim} | Freq Dim: {self.freq_feat_dim}")
        print(f"[Fusion Info] Total Fusion Dimension: {total_dim}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),  # LayerNorm 对多模态特征拼接更稳健
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # --- Stream 1: Time ---
        t_feat = self.time_extractor(x)  # (B, time_shared_nf, L)
        t_vec = self.time_pool(t_feat).squeeze(-1)  # (B, time_shared_nf)

        # --- Stream 2: Frequency ---
        f_feat = self.freq_extractor(x)  # (B, base*4, L)
        f_feat = self.freq_bottleneck(f_feat)  # (B, bottleneck, L)

        # 混合池化
        f_cov = self.freq_group_cov(f_feat)  # (B, dim_cov)
        f_mean = self.freq_gap(f_feat).squeeze(-1)  # (B, dim_mean)
        f_vec = torch.cat([f_cov, f_mean], dim=1)  # (B, dim_total_freq)

        # --- Fusion ---
        combined = torch.cat([t_vec, f_vec], dim=1)

        return self.classifier(combined)


class TimeDomain_Ablation_Model(nn.Module):
    """
    支持层级消融实验的时域模型
    layers_mode:
      - '1': 仅使用第一层卷积提取的浅层特征
      - '1+2': 使用两层卷积提取的深层特征 (未融合)
      - '1+2+3': 完整模型 (两层卷积 + 深度可分离融合层)
    """

    def __init__(self, num_classes=8, num_imu=9, num_strain=2,
                 base_nf=16, shared_nf=64, dropout=0.1, layers_mode='1+2+3'):
        super(TimeDomain_Ablation_Model, self).__init__()

        self.layers_mode = layers_mode
        self.num_classes = num_classes

        # 1. 实例化核心骨干网络
        # 注意：这里我们复用 MultiBranch_Time_Net，但会在 forward 中手动控制流向
        self.backbone = MultiBranch_Time_Net(
            num_imu, num_strain,
            imu_nf=base_nf,
            strain_nf=base_nf // 2,
            shared_nf=shared_nf,
            dropout=dropout
        )

        # 2. 动态计算分类器的输入维度 (根据消融模式)
        if layers_mode == '1':
            # Layer 1 输出: IMU_b1 (base_nf) + Strain_b1 (max(4, base_nf//4))
            # 这里的计算逻辑必须与 MultiBranch_Time_Net 内部一致
            strain_start_dim = max(4, (base_nf // 2) // 2)
            self.feat_dim = base_nf + strain_start_dim

        elif layers_mode == '1+2':
            # Layer 2 输出: IMU_b2 (base_nf*2) + Strain_b2 (base_nf)
            # 注意: Strain_b2 输出是 strain_nf * 2，即 (base_nf//2) * 2 = base_nf
            self.feat_dim = (base_nf * 2) + base_nf

        elif layers_mode == '1+2+3':
            # Layer 3 输出: Shared Block 输出 (shared_nf)
            self.feat_dim = shared_nf

        else:
            raise ValueError(f"Unknown layers_mode: {layers_mode}")

        print(f"[Time-Ablation] Mode: {layers_mode} | Classifier Input Dim: {self.feat_dim}")

        # 3. 池化层 (将时间维度 L 压缩为 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x shape: (B, L, C) -> 需要转置并切片
        x_i = x[:, :, :9].transpose(1, 2)
        x_s = x[:, :, 9:].transpose(1, 2)

        feat_i = self.backbone.imu_b1(x_i)
        feat_s = self.backbone.strain_b1(x_s)

        if self.layers_mode == '1':
            feat = torch.cat([feat_i, feat_s], dim=1)
            feat = self.pool(feat)
            return self.classifier(feat)

        feat_i = self.backbone.imu_b2(feat_i)
        feat_s = self.backbone.strain_b2(feat_s)

        if self.layers_mode == '1+2':
            feat = torch.cat([feat_i, feat_s], dim=1)
            feat = self.pool(feat)
            return self.classifier(feat)

        cat_feat = torch.cat([feat_i, feat_s], dim=1)
        feat = self.backbone.shared(cat_feat)
        feat = self.backbone.bn(feat)
        feat = self.backbone.act(feat)
        feat = self.backbone.drop(feat)

        if self.layers_mode == '1+2+3':
            feat = self.pool(feat)
            return self.classifier(feat)


class Baseline_SimpleCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=11, seq_len=100, base_nf=16, dropout=0.1):
        super(Baseline_SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Layer 1: 普通卷积
            nn.Conv1d(input_channels, base_nf, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_nf),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64

            # Layer 2: 普通卷积
            nn.Conv1d(base_nf, base_nf * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_nf * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 -> 32

            # Layer 3: 普通卷积
            nn.Conv1d(base_nf * 2, base_nf * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_nf * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Pool
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_nf * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 128, 11) -> 需要转置为 (B, 11, 128) 适配 Conv1d
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)