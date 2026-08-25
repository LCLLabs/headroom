# CoACT HTTP `/prune` 服务调用文档

本文档面向**外部调用方**，说明如何通过 HTTP 接口调用 CoACT 代码/工具输出压缩服务。

---

## 1. 服务说明

CoACT 提供与 SWE-Pruner 类似的 HTTP 接口，用于压缩 agent 工具输出或源码片段，在保留关键信息的同时减少 token 消耗。

| 项目 | 说明 |
|------|------|
| 协议 | HTTP/JSON |
| 默认地址 | `http://<host>:8002` |
| 主要接口 | `GET /health`、`POST /prune` |
| 认证 | 默认无需 API Key（内网部署） |

> **注意**：8002 是 HTTP 包装层端口；底层 vLLM 推理服务（8001）由运维侧单独启动，调用方无需直接访问。

---

## 2. 快速开始

### 2.1 健康检查

```bash
curl http://<host>:8002/health
```

**响应示例：**

```json
{
  "status": "healthy",
  "backend_reachable": true,
  "backend_endpoint": "http://localhost:8001/v1"
}
```

- `backend_reachable: true` 表示后端模型服务正常，可以调用 `/prune`。
- 若为 `false`，请联系服务管理员检查 vLLM 是否已启动。

### 2.2 最简压缩调用

```bash
curl -X POST http://<host>:8002/prune \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the exact pytest error message?",
    "code": "FAILED tests/test_utils.py::test_add\nAssertionError: assert 3 == 4\n1 failed in 0.12s"
  }'
```

---

## 3. API 参考

### 3.1 `GET /health`

检查服务是否存活，以及后端推理服务是否可达。

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"healthy"` |
| `backend_reachable` | boolean | 后端 vLLM 是否可达 |
| `backend_endpoint` | string | 后端 vLLM 地址（运维信息） |

---

### 3.2 `POST /prune`

对一段文本进行语义压缩。

**请求头：**

```
Content-Type: application/json
```

#### Request Body

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 聚焦问题。说明你需要从 `code` 中保留什么信息，越具体压缩效果越好 |
| `code` | string | ✅ | — | 待压缩的原始内容（工具输出、源码、日志、grep 结果等） |
| `goal` | string | ❌ | `""` | 全局任务目标，例如 PR 描述或 bug 修复目标 |
| `tool_call` | string | ❌ | `null` | 产生该输出的命令，例如 `"cat src/utils.py"` |

#### Response Body

| 字段 | 类型 | 说明 |
|------|------|------|
| `pruned_code` | string | **压缩后的文本**，调用方应直接使用此字段 |
| `origin_token_cnt` | integer | 原始 `code` 的 token 数 |
| `left_token_cnt` | integer | 剪后 `pruned_code` 的 token 数 |
| `model_input_token_cnt` | integer | 发给模型的 prompt token 数 |
| `kept_frags` | integer[] | 保留的原始行号（1-indexed）；`code` 类型时有效 |
| `compression_type` | string | 压缩类型，见下表 |
| `error_msg` | string \| null | 错误信息；成功时为 `null` |

#### `compression_type` 取值

| 值 | 含义 |
|----|------|
| `unchanged` | 未压缩，内容与原始 `code` 相同 |
| `plain` | 纯文本摘要，适用于日志、测试输出等非代码内容 |
| `code` | 按行保留/折叠，适用于源码、diff、grep 结果等 |
| `invalid` | 模型输出无效或后端调用失败，已回退为原始 `code` |

#### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| `200` | 请求成功（即使压缩失败，`error_msg` 中会说明，并回退原始内容） |
| `422` | 请求参数不合法（如 `query` 或 `code` 为空） |
| `500` | 服务内部错误 |

---

## 4. 调用示例

### 4.1 curl — 压缩 Python 源码

```bash
curl -X POST http://<host>:8002/prune \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "query": "What is the current implementation of validate() and what does it return for empty string input?",
  "goal": "Fix the bug: validate() should raise ValueError when input is empty string.",
  "tool_call": "sed -n '1,80p' src/utils.py",
  "code": "# Copyright 2024 Example Corp\n\nimport re\nfrom typing import Any\n\ndef normalize(text: str) -> str:\n    return text.strip().lower()\n\ndef slugify(text: str) -> str:\n    text = normalize(text)\n    return re.sub(r\"[^a-z0-9]+\", \"-\", text).strip(\"-\")\n\ndef validate(value: str) -> str:\n    if value is None:\n        raise TypeError(\"value must be a string\")\n    if not isinstance(value, str):\n        raise TypeError(\"value must be a string\")\n    return value.strip()\n\ndef parse_config(raw: dict[str, Any]) -> dict[str, Any]:\n    if not isinstance(raw, dict):\n        raise ValueError(\"config must be a dict\")\n    return {\"name\": raw.get(\"name\", \"default\")}\n\ndef format_user(user: dict[str, Any]) -> str:\n    return f\"{user.get('id')}:{user.get('name', 'unknown')}\"\n\ndef checksum(data: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(data).hexdigest()"
}
EOF
```

**响应示例：**

```json
{
  "pruned_code": "(compressed 19 lines: imports and helper functions)\ndef validate(value: str) -> str:\n    if value is None:\n        raise TypeError(\"value must be a string\")\n    if not isinstance(value, str):\n        raise TypeError(\"value must be a string\")\n    return value.strip()\n\n(compressed 47 lines: other utility functions)",
  "origin_token_cnt": 511,
  "left_token_cnt": 78,
  "model_input_token_cnt": 1368,
  "kept_frags": [20, 21, 22, 23, 24, 25, 26],
  "error_msg": null,
  "compression_type": "code"
}
```

上例中 token 从 **511 → 78**，压缩率约 **85%**。

### 4.2 curl — 仅提取压缩结果

```bash
curl -s -X POST http://<host>:8002/prune \
  -H "Content-Type: application/json" \
  -d '{"query":"find validate()","code":"def foo(): pass\ndef validate(x): return x\n"}' \
  | jq -r '.pruned_code'
```

### 4.3 Python

```python
import httpx

PRUNE_URL = "http://<host>:8002/prune"

def compress_code(code: str, query: str, goal: str = "", tool_call: str | None = None) -> str:
    response = httpx.post(
        PRUNE_URL,
        json={
            "query": query,
            "code": code,
            "goal": goal,
            "tool_call": tool_call,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error_msg"):
        raise RuntimeError(result["error_msg"])
    return result["pruned_code"]


# 示例
pruned = compress_code(
    code=open("src/utils.py").read(),
    query="What is validate() and what does it return for empty input?",
    goal="Fix empty string validation",
    tool_call="cat src/utils.py",
)
print(pruned)
```

### 4.4 JavaScript (fetch)

```javascript
const response = await fetch("http://<host>:8002/prune", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What is the pytest failure reason?",
    code: toolOutput,
  }),
});
const result = await response.json();
const compressed = result.pruned_code;
```

---

## 5. 输出格式说明

### 5.1 `code` 类型（源码类）

无关行会被折叠为占位符，关键行保留原文：

```text
(compressed 19 lines: imports and helper functions)
def validate(value: str) -> str:
    if value is None:
        raise TypeError("value must be a string")
    return value.strip()
(compressed 47 lines: other utility functions)
```

### 5.2 `plain` 类型（日志/测试类）

返回一段精简后的自然语言摘要：

```text
3 tests passed; no failure traceback was present.
```

### 5.3 `unchanged` 类型

`pruned_code` 与原始 `code` 完全相同，表示模型判断无需压缩。

---

## 6. 使用建议

### 6.1 如何写好 `query`

`query` 是影响压缩质量的最重要参数。

| 推荐 ✅ | 不推荐 ❌ |
|---------|-----------|
| `"find validate() signature and exception for empty input"` | `"read the file"` |
| `"what line does pytest report as the failure location?"` | `"understand the code"` |
| `"show the stack trace frame inside utils.py"` | `"show me everything"` |

### 6.2 何时会有明显压缩效果

- `code` 较长（数百 token 以上）
- `query` 聚焦局部信息（某个函数、某段错误、某个类）
- 内容中存在大量与 `query` 无关的部分

短内容或 `query` 过于宽泛时，可能返回 `compression_type: "unchanged"`。

### 6.3 错误处理

```python
result = response.json()

if result["error_msg"]:
    # 后端失败或模型输出无效，pruned_code 已回退为原始 code
    print(f"Compression failed: {result['error_msg']}")
    use_output = result["code"]  # 实际上 pruned_code == 原始 code
else:
    use_output = result["pruned_code"]

# 可选：查看压缩统计
print(f"Tokens: {result['origin_token_cnt']} -> {result['left_token_cnt']}")
print(f"Type: {result['compression_type']}")
```

---

## 7. 运维信息（供管理员参考）

调用方通常无需关心以下内容；服务不可用时请联系管理员。

### 7.1 启动顺序

```bash
# 1. 启动 vLLM 推理服务（端口 8001）
bash start_CoACT.sh

# 2. 启动 HTTP 包装服务（端口 8002）
bash start_CoACT_http.sh
```

### 7.2 默认端口

| 服务 | 端口 | 说明 |
|------|------|------|
| vLLM | 8001 | 内部推理，OpenAI 兼容 API |
| HTTP `/prune` | 8002 | **外部调用入口** |

### 7.3 日志

| 文件 | 内容 |
|------|------|
| `CoACT.log` | vLLM 推理日志 |
| `CoACT_http.log` | HTTP 服务日志 |

---

## 8. 常见问题

**Q: 需要 API Key 吗？**  
A: 默认不需要。若部署方启用了网关鉴权，请向其索取访问方式。

**Q: 请求超时怎么办？**  
A: 长文本压缩可能需要数秒到数十秒。建议客户端 `timeout` 设为 **120 秒**以上。

**Q: `left_token_cnt` 是什么？**  
A: 压缩后 `pruned_code` 的 token 数，用于衡量压缩效果。不是模型生成的 JSON 长度。

**Q: 和 SWE-Pruner 有什么区别？**  
A: 接口风格类似（`POST /prune`，传 `query` + `code`），但 CoACT 使用 LLM 语义压缩，SWE-Pruner 使用专用裁剪模型。两者可独立替换使用。

**Q: 服务返回 `backend_reachable: false`？**  
A: HTTP 层正常，但后端 vLLM 未启动。请联系管理员执行 `bash start_CoACT.sh`。

---

## 9. 接口摘要

```
GET  http://<host>:8002/health          # 健康检查
POST http://<host>:8002/prune           # 压缩
```

**最小请求体：**

```json
{
  "query": "<你的聚焦问题>",
  "code": "<待压缩的原始文本>"
}
```

**关键响应字段：**

```json
{
  "pruned_code": "<压缩后的文本，直接使用>",
  "origin_token_cnt": 511,
  "left_token_cnt": 78,
  "compression_type": "code",
  "error_msg": null
}
```
