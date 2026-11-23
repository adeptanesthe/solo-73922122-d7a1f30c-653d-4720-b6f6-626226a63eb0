# 风机耗电共用模型平台（Fan Power Model Platform）

各厂只上报工况摘要（样本数 + 整数向量和 + 外积和 + 压差零点修正量），平台用
NumPy 岭回归拟合一套共用的"转速/压差 → 每分钟耗电量"模型，按留一厂交叉验证
选 λ，SQLite 多版本保存，支持凭版号原子替换单厂修正量。

## 快速开始

```bash
docker compose up --build      # 服务监听 :8000
./demo.sh                      # 演示：建厂 → 预测 → 更正修正量 → 冲突
```

本地开发（无 Docker）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn --factory app.main:create_app --port 8000
.venv/bin/python -m pytest tests/ -q     # 26 个自测
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/versions` | 上报 3–8 家厂的摘要，训练并发布新版本（201） |
| GET | `/versions/current` | 现行版：所选 λ、逐厂误差、修正量、模型 |
| GET | `/versions/{id}` | 历史版本（不可变） |
| POST | `/versions/{id}/correction` | 凭版号替换一厂修正量；成功则修正量+模型+逐厂误差作为新版本一起生效；版号冲突 409 / 重算失败 422，均保留现行版 |
| POST | `/predict` | 按厂号 + 原始转速、压差预测耗电，返回所用版号；重算期间沿用现行版 |
| GET | `/health` | 健康检查 |

### 上报格式（`POST /versions`）

```json
{
  "factories": [
    {
      "factory_id": "F1",
      "n": 1440,
      "sums":  [Σ转速, Σ压差, Σ耗电],
      "outer": [[Σs², Σsp, Σsy], [Σsp, Σp², Σpy], [Σsy, Σpy, Σy²]],
      "correction": 0
    }
  ]
}
```

`sums`/`outer` 均为**整数**（任意精度，JSON 大整数无损解析）；`correction`
为加到压差上的整数零点修正量。预测时传原始读数，平台自动加上该厂修正量。

## 关键设计

- **大基值方差保留**：协方差分子 `N·Σxxᵀ − ΣxΣxᵀ` 全程用 Python 大整数精确
  计算，最后一步才转 float64。读数基值 1e12、方差仅个位数时，朴素
  `E[x²]−E[x]²` 会被灾难性抵消成 0，这里仍精确保留（有专门测试）。
  入库前对所有厂做精确 PSD 校验（全部主子式 ≥ 0，整数判定）。
- **岭回归**：损失 = 均方误差 + λ·‖w‖²，截距免罚。特征按训练厂的 pooled
  均值与**总体标准差**标准化后，闭式解 `(C + λI)w = Cov(z,y)`，
  `b = ȳ`；零方差特征忽略。
- **留一厂验证**：轮流留一厂，只用其余厂的均值/总体标准差标准化并拟合，
  评估留出厂 MSE（由充分统计量精确求出，无需原始行）。在
  {0.01, 0.1, 1} 中选"各留出厂 MSE 最大值"最小的 λ，并列取较小者，
  再用全厂重训。
- **版本原子性**：修正量、模型、逐厂误差在同一 SQLite 事务写入，最后才翻动
  `current_version` 指针；预测只读已提交的现行版，因此重算期间预测仍沿用
  现行版。凭版号更新时若版号已过期 → 409 且现行版不变；重算失败同样保留。
- **并发**：WAL 模式 + 写锁；计算在事务外完成，提交窗口极短。

## 演示（demo.sh）

`scripts/make_demo_data.py` 生成确定性的 4 厂数据（转速基值 1e6、压差基值
5e5 的原始计数；F3 压差零点漂移 +500 但按 0 上报）。`demo.sh` 依次：

1. `POST /versions` 建厂 → F3 留厂误差 33.3 明显偏高，λ=0.01；
2. `POST /predict` 用原始读数预测 F3；
3. `POST /versions/1/correction` 把 F3 修正量改为 500 → 版本 2，
   逐厂误差全部降到 ≈1–2，模型与修正量同时生效；
4. 再次预测 → 结果改变且 `version=2`；
5. 用过期版号重放 → HTTP 409，现行版保持版本 2。

## 剩余事项

- 本环境无 Docker，`docker compose` 未实际构建过；Dockerfile 为标准
  python:3.12-slim 流程，建议在有 Docker 的机器上跑一次 `docker compose up --build`。
- 单进程服务（uvicorn 单 worker）；多副本部署需把 SQLite 换成集中式存储。
- 未做鉴权/审计；生产环境应加认证与操作日志。
- 留一厂验证对工况迥异的厂会外推放大误差（属预期行为）；如需可扩展分组验证。
