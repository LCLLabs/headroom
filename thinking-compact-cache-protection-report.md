# thinking 压缩误伤 prompt cache 的修复

## 背景

三个方向的灰度修复(`87b04b03`)上线后,`HEADROOM_THINKING_COMPACT` 对 `deepseek/glm/qwen/minimax/kimi` 等系列真正生效(此前因为 `bills_prior_thinking` 的版本号解析 bug,这段逻辑对这些系列基本是空转)。排查命中率下降问题时发现:`compact_thinking_to_text` 本身的窗口设计,会在**每一轮**都主动制造一次可以避免的 prompt cache miss,和这次生效的范围是同一批流量,时间线吻合。

## 问题机制

`compact_thinking_to_text(..., keep_last_turns=N)` 用"离对话末尾多少轮"这个**相对位置**来决定哪些 thinking block 保留原文、哪些压缩成摘要。对话每往前推进一轮,这个边界就整体后移一格——上一轮还被保护、原样转发的那条 thinking,这一轮就会被划到边界外,第一次被压缩,字节从 `thinking`(原文)变成 `text`(压缩摘要)。

Provider 端的 prompt cache 按字节前缀哈希比对,前缀里任意一处字节变化,那个位置之后的全部缓存立即失配。也就是说:只要这个开关开着,**每一轮都会在原本可以维持热缓存的地方,主动切掉一次缓存**——不是偶发的边界情况,是这套"数轮次"设计本身必然、持续发生的副作用。

## 修复思路

不去掉压缩,只改"允许压缩边界推进的依据"。

代理请求路径里已经有一个更权威的值:`frozen_message_count`(`headroom/proxy/handlers/anthropic.py`,来自 `PrefixCacheTracker.get_frozen_message_count()`)。这个值不是猜的,是上一轮 Anthropic 响应里**真实的 `cache_read_input_tokens`** 反推出来的——"到第几条消息为止,现在确实是从 provider 缓存里读出来的"。代理内部其它所有压缩/裁剪环节(`content_router.py` 等)早就在遵守同一个规则:`index < frozen_message_count` 的消息一律不碰。

这次把 `compact_thinking_to_text` 也接入了同一条规则,而不是引入新的开关或新的状态跟踪:

- **`index < frozen_message_count`(确认还在被缓存读取的区域)**:无论 `keep_last_turns` 怎么算,都不压缩,原样转发。避免主动切热缓存。
- **`index >= frozen_message_count`(上一轮真实反馈里已经不在被读取的区域)**:按原来的 `keep_last_turns` 逻辑正常压缩。这部分缓存反正没有在被读取,压缩不额外产生代价。

## 改动

**`headroom/transforms/thinking_compactor.py`** —— `compact_thinking_to_text()` 新增参数 `frozen_message_count: int = 0`(默认 0,即不额外保护,老行为不变),在跳过条件里加入 `or i < frozen_message_count`。

```python
def compact_thinking_to_text(
    messages: list[dict[str, Any]],
    *,
    kompress: Any,
    keep_last_turns: int = 1,
    min_words: int = 40,
    frozen_message_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ...
    for i, m in enumerate(messages):
        content = m.get("content")
        if (
            m.get("role") != "assistant"
            or i in keep
            or i < frozen_message_count
            or not isinstance(content, list)
            or not any(isinstance(b, dict) and b.get("type") == "thinking" for b in content)
        ):
            out.append(m)
            continue
```

**`headroom/proxy/handlers/anthropic.py`** —— 调用点复用函数体内已经算好、并被其它所有下游转换共享的同一个 `frozen_message_count` 局部变量(该变量在 1791 行处从 `_prep.frozen_message_count` 取得,已经过 pruning/裁剪造成的位置漂移校正,是"权威的当前位置真值",不需要重新计算):

```python
_tc_messages, _tc_stats = compact_thinking_to_text(
    body.get("messages"),
    kompress=_tc_kompress,
    keep_last_turns=_tc_keep,
    min_words=40,
    frozen_message_count=frozen_message_count,
)
```

没有新增环境变量:`frozen_message_count=0` 是安全默认值,行为等价于修复前;真正生效仍然完全受既有的 `HEADROOM_THINKING_COMPACT` 灰度开关控制,不引入新的开关面。

## 测试

`tests/test_thinking_compactor.py` 新增 `test_frozen_message_count_protects_cached_prefix`:构造一段 `keep_last_turns=0`(本应全部压缩)的对话,传入 `frozen_message_count=2`,断言边界前的 thinking block 保持原文、边界后的正常压缩,`turns_compacted == 1`(而不是 2)。

```
pytest tests/test_thinking_compactor.py -q   # 26 passed
```

（`headroom/proxy/handlers/anthropic.py` 因缺少编译好的 `headroom._core` 扩展,当前环境无法跑集成测试;这是既有的环境问题,和本次改动无关,已用 `py_compile` 确认改动本身没有语法/引用错误。）

## 效果与局限

- **解决的**:同一批因 family 修复而"真正开始生效"的流量,不会再在压缩之外额外制造一次每轮必发生的 cache miss。压缩收益不变,只是不再拿"确认还热"的那部分缓存去换。
- **没解决的(留给下一步)**:`frozen_message_count` 只保护"上一轮确认还热"的区域,不能让"确认已经冷"的区域拿到更彻底的收益——之前讨论过的"冷前缀直接丢弃整段 thinking,而不只是压缩成摘要"(对齐 `compact_reasoning_openai_chat` 已有的 `drop=True` 冷启动钩子)还没有对应实现到 `compact_thinking_to_text` 里,是这次改动之外、值得单独跟进的下一版本。
