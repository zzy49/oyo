# 图1 本文方法总览

---

## 区域A：总体管线（6阶段横向流程）

```mermaid
graph LR
    INPUT["📥 输入：单目内窥镜 RGB 帧序列"]

    INPUT --> S1

    S1["① 深度估计<br/>ResNet-18 + MotionEncoder<br/>+ DepthDecoder"]
    S2["② 特征匹配<br/>LoFTR Transformer<br/>帧间 2D 对应点"]
    S3["③ 自适应静止点筛选<br/>3D位移 + 自适应阈值<br/>Δd ≤ median + 1.0×std"]
    S4["④ EPnP + 尺度恢复<br/>pnp_scale 零配置<br/>λ = 0.5mm / median_step"]
    S5["⑤ 深度一致性校正<br/>scale_corr 微调<br/>depth × scale_corr"]
    S6["⑥ 组织运动解耦<br/>链式追踪<br/>世界坐标"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6

    S3 -.-> FORMULA1["τ = median(Δd) + 1.0×std(Δd)"]
    S4 -.-> FORMULA2["★ 零配置核心"]

    style INPUT fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style S1 fill:#e3f2fd,stroke:#1976d2
    style S2 fill:#e8eaf6,stroke:#5c6bc0
    style S3 fill:#f3e5f5,stroke:#ab47bc
    style S4 fill:#fff3e0,stroke:#ef6c00
    style S5 fill:#fce4ec,stroke:#d81b60
    style S6 fill:#e8f5e9,stroke:#43a047
```

---

## 区域B：深度估计网络结构（编码器-解码器）

```mermaid
graph LR
    subgraph ENCODER["编码器 ResNet-18"]
        direction TB
        CUR["RGB H×W×3"]
        C0["Conv1+MaxPool<br/>64ch H/4"]
        L1["Layer1<br/>64ch H/4"]
        L2["Layer2<br/>128ch H/8"]
        L3["Layer3<br/>256ch H/16"]
        L4["Layer4<br/>512ch H/32"]
        CUR --> C0 --> L1
        L1 --> L2 --> L3 --> L4
    end

    subgraph MOTION["MotionEncoder 仅Ours"]
        direction TB
        CAT["当前帧+相邻帧 6通道"]
        M1["Conv1 7×7<br/>6→32 H/2"]
        M2["Conv2 5×5<br/>32→64 H/4"]
        M3["Conv3 3×3<br/>64→64 H/8"]
        M4["Conv4 3×3<br/>64→128 H/16"]
        M5["Conv5 3×3<br/>128→256 H/32"]
        CAT --> M1 --> M2 --> M3 --> M4 --> M5
    end

    subgraph DECODER["解码器 U-Net"]
        direction BT
        D4["Upconv4<br/>256ch H/16"]
        D3["Upconv3<br/>128ch H/8"]
        D2["Upconv2<br/>64ch H/4"]
        D1["Upconv1<br/>32ch H/2"]
        D0["Upconv0<br/>16ch H"]
        D4 --> D3 --> D2 --> D1 --> D0
    end

    subgraph HEAD["混合深度头"]
        direction TB
        BIN["分类头<br/>64-bin Logits"]
        RES["残差头<br/>tanh Residual"]
        DB["depth_from_bins()"]
        OUT["深度图 D̂<br/>1–500mm"]
        D0 --> BIN --> DB --> OUT
        D0 --> RES --> DB
    end

    L4 -->|"feature[4] 512ch"| D4
    L3 -.->|"skip + 门控"| D3
    L2 -.->|"skip + 门控"| D2
    L1 -.->|"skip + 门控"| D1
    L1 -.->|"skip + 门控"| D0

    M2 -->|"mot[0] 64ch"| D2
    M3 -->|"mot[1] 64ch"| D1
    M4 -->|"mot[2] 128ch"| D1
    M5 -->|"mot[3] 256ch"| D0

    HEAD --> FORMULA_D["D = Σ pᵢ·bᵢ + R·Δb/2<br/>bₖ = exp(ln1.0 + k/63·ln500)"]

    style ENCODER fill:#e3f2fd,stroke:#1976d2
    style MOTION fill:#fce4ec,stroke:#d81b60
    style DECODER fill:#e8f5e9,stroke:#43a047
    style HEAD fill:#fff9c4,stroke:#f9a825
```

---

## 区域C：三路输出

```mermaid
graph LR
    DEPTH_SRC["深度预测图"] --> OUT1["📤 输出① 深度图<br/>.npz  1–500mm<br/>192×192"]
    POSE_SRC["相机绝对位姿"] --> OUT2["📤 输出② 相机位姿<br/>.npy  4×4矩阵<br/>单位: mm"]
    MOTION_SRC["运动组织位姿"] --> OUT3["📤 输出③ 组织运动<br/>.json  逐track<br/>世界坐标, mm"]

    style OUT1 fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style OUT2 fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px
    style OUT3 fill:#e1bee7,stroke:#ab47bc,stroke-width:2px
```

---

## 关键公式标注

| 位置 | 公式 |
|------|------|
| 区域A 阶段③ | \( \tau = \text{median}(\Delta d) + 1.0 \times \text{std}(\Delta d) \) |
| 区域A 阶段④ | \( \lambda = \frac{0.5\text{mm}}{\text{median}(\text{PnP\_step})} \) ★ 零配置核心 |
| 区域B 输出头 | \( D = \sum p_i \cdot b_i + R \cdot \frac{\Delta b}{2} \) |
| 区域B bin分布 | \( b_k = \exp(\ln 1.0 + \frac{k}{63} \cdot \ln 500), \quad k = 0,1,...,63 \) |

---

## 性能指标（EndoSLAM c1_transverse1_t1_v2）

| 指标 | Ours | Monodepth2 | ManyDepth | Lite-Mono |
|------|------|-----------|-----------|-----------|
| ATE ↓ | **5.08 mm** | 7.15 mm | 9.18 mm | 7.84 mm |
| RPE-T ↓ | 0.39 mm | 0.27 mm | 0.31 mm | **0.13 mm** |
| RPE-R ↓ | **0.43°** | 2.44° | 2.06° | 2.08° |
| Scale Err ↓ | **24.97%** | 159.22% | 54.89% | 198.13% |
| 成功率 ↑ | **100%** | 96.6% | 93.1% | 88.8% |

---

> **图1 本文方法总览**
> Fig.1 Overview of the proposed method

渲染工具：VS Code (Mermaid插件) 或 https://mermaid.live
