# Gradient Inversion Attack 상황에서 강인한 Transformer Family 구조 탐색


| 모델명 | 크기 | 파라미터 수 | Top-1 정확도 |
|---|---|---|---|
| **ResNet** | ResNet-50 | 25M | 76-78% |
|  | ResNet-152 | 60M | 78-80% |
| **MLP-Mixer** | MLP-Mixer-B/16 | 59M | 76% |
|  | MLP-Mixer-L/16 | 207M | 78% |
| **ViT** | ViT-B/16 | 86M | 77-84% |
|  | ViT-L/16 | 307M | 80-87% |
| **Swin Transformer** | Swin-T | 29M | 81.3% |
|  | Swin-B | 88M | 83.5% |
| **FNet** | FNet-B/16 | 75M | 75-77% |
|  | FNet-L/16 | 300M | 78-80% |
| **Longformer** | Longformer-Base | 149M | 79-81% |
|  | Longformer-Large | 435M | 81-83% |
