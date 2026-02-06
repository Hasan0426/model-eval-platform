这是一个非常值得记录的技术复盘（Post-mortem）。这个问题涉及到了 **Pandas（数据处理）**、**JSON（数据交换）** 和 **PostgreSQL（数据存储）** 三者之间的底层标准冲突，是数据工程中极易踩的坑。

以下是详细的 Bug 记录文档，建议保存在 `docs/troubleshooting_log.md` 或作为你的开发周记。

---

# 🐛 Bug 记录：PostgreSQL JSON 字段拒绝 "NaN" 值导致 Worker 崩溃

**日期**：2023-10-XX
**模块**：Worker (Profiling Service)
**严重级别**：Critical (导致任务失败，数据无法写入数据库)

## 1. 问题描述 (Symptoms)

在上传某些特定 CSV 文件（包含缺失值或单行数据）时，API 响应正常（状态 PENDING），但后台 Celery Worker 任务失败。

**报错日志核心信息**：

```text
sqlalchemy.exc.DataError: (psycopg2.errors.InvalidTextRepresentation) invalid input syntax for type json
DETAIL:  Token "NaN" is invalid.
CONTEXT:  JSON data, line 1: {"age": {"mean": 56.0, "std": NaN...

```

**现象**：

* 数值列的统计结果（如标准差 `std`）在样本不足时计算出了 `NaN`。
* 预览数据（Preview）中的缺失值在 Pandas 中表现为 `NaN`。
* Worker 试图将包含 `NaN` 的字典写入 Postgres 的 `JSONB` 字段时被拒绝。

## 2. 根本原因 (Root Cause Analysis)

### A. 标准冲突

* **Pandas/Numpy**：使用 IEEE 754 标准的浮点数 `NaN` (Not a Number) 来表示缺失数据或无效计算结果。
* **JSON 标准 (RFC 8259)**：JSON 格式**不支持** `NaN`, `Infinity`, `-Infinity`。它只支持 `null`。
* **PostgreSQL**：其 `JSON` 和 `JSONB` 数据类型严格遵循 JSON 标准。如果输入的字符串包含 `NaN`，Postgres 会认为这是无效的 JSON 语法，直接抛出异常。

### B. Python 的误导

Python 自带的 `json` 库默认是**宽容**的。

```python
import json
json.dumps(float('nan')) 
# 输出: 'NaN'  <-- 这不是合法的 JSON，但 Python 默认允许输出了

```

所以我们在本地测试打印 JSON 时看起来没问题，但传给 Postgres 就挂了。

### C. Pandas 的类型固执 (The Trap)

我们曾尝试在 DataFrame 内部清洗数据：

```python
# ❌ 失败的尝试
df.replace({np.nan: None}) 

```

**为什么失败？**
如果一列数据是 `float` 类型（例如 `age` 列），Pandas 会强制维护其数据类型一致性。Python 的 `None` 不是 float，所以 Pandas 会**自动把 `None` 变回 `NaN**` 以保持该列为 float 类型。这导致脏数据无法在 DataFrame 层面彻底清除。

## 3. 解决方案 (Solution)

**策略**：放弃在 Pandas 内部清洗，改为**后处理（Post-processing）**。在数据转换为 Python 原生字典/列表后，使用递归函数强制清洗。

**修复代码 (`core/profile.py`)**：

引入全局清洗函数 `clean_nan`：

```python
import math
import numpy as np
import pandas as pd

def clean_nan(obj):
    """
    递归遍历字典或列表，将所有的 NaN / Infinity / NaT 强制转换为 None。
    这是唯一能 100% 保证 JSON 兼容的方法。
    """
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None # JSON null
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj) # 顺便解决 numpy int 不兼容问题
    elif obj is pd.NA or obj is np.nan:
        return None
    return obj

```

并在所有返回给数据库的数据上应用此函数：

```python
# 例如在 infer_schema 函数末尾
return clean_nan({
    "rows": n_rows,
    "columns": columns,
    ...
})

```

## 4. 经验总结 (Key Takeaways)

1. **不要相信 Pandas 的 `replace(nan, None)**`：只要列类型是 float，它基本都会变回去。
2. **JSON 标准很严格**：不要以为 Python 能 dump 出来的就是合法 JSON，数据库通常比 Python 更严格。
3. **清洗时机**：数据清洗最好发生在 **[Pandas -> Python Dict]** 转换之后，**[Python Dict -> JSON String]** 转换之前。
4. **Numpy 类型陷阱**：除了 `NaN`，`np.int64` 和 `np.float32` 常常也不能直接被 JSON 序列化，最好在清洗函数里一并转为 Python 原生的 `int` 和 `float`。







第二个bug
### 📝 Bug 记录：XGBoost 维度不匹配错误

#### 1. 错误现象 (Symptom)

* **报错信息**:
```text
xgboost.core.XGBoostError: Check failed: preds.Size() == info.labels.Size() (600 vs. 200) : label and prediction size not match, hint: use merror or mlogloss for multi-class classification

```


* **发生场景**: Worker 执行 `task_train_model` 训练任务时。
* **触发条件**: 上传的数据集目标列（如 `gender`）包含超过 2 个类别（例如 3 个），但代码强制使用了二分类模式。

#### 2. 详细原因 (Root Cause)

* **维度冲突分析**:
* **200**: 验证集样本数（Label 数量）。
* **600**: 模型输出的预测值数量。
* **600 / 200 = 3**: 说明 XGBoost 自动识别出数据有 **3 个类别**，因此为每个样本输出了 3 个概率值（Softmax 输出）。


* **配置错误**:
* XGBoost 检测到数据是多分类（Multi-class），自动调整了输出形状。
* 但在 `core/ml.py` 代码中，参数被**硬编码**为二分类模式：
* `objective`: `"binary:logistic"` (仅适用于二分类)
* `eval_metric`: `"logloss"` (仅适用于二分类)


* **结果**: 模型输出的形状（3列）与目标函数期望的形状（1列）不匹配，导致 CUDA/C++ 底层校验失败。



#### 3. 解决方案 (Resolution)

修改 `core/ml.py` 中的 `train_xgboost_baseline` 函数，增加**自动检测类别数量**的逻辑，实现“二分类”与“多分类”的动态切换。

**核心代码变更**:

1. **特征预处理**: 强制对 Target 列进行 `LabelEncoder` 编码，并计算唯一类别数 (`num_classes`)。
2. **动态参数配置**:
```python
# 伪代码逻辑
if num_classes > 2:
    # 多分类模式
    params["objective"] = "multi:softprob"
    params["eval_metric"] = "mlogloss"
    params["num_class"] = num_classes
else:
    # 二分类模式
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"

```


3. **部署**: 修改代码后，执行 `docker compose ... restart worker` 重启 Worker 服务以加载新逻辑。