from openai import OpenAI

# ===== 选一个你喜欢的平台，取消注释即可 =====

# ECNU Chat（校内平台，推荐首选）
client = OpenAI(api_key="sk-e7a80a162dcc4e0ca6a68cc7527459ac", base_url="https://chat.ecnu.edu.cn/open/api/v1")
model = "ecnu-max"  # 也可用 ecnu-plus（支持图片）、ecnu-turbo（速度快）

# Kimi（月之暗面）
# client = OpenAI(api_key="你的key", base_url="https://api.moonshot.cn/v1")
# model = "moonshot-v1-auto"

# 智谱 GLM
# client = OpenAI(api_key="你的key", base_url="https://open.bigmodel.cn/api/paas/v4")
# model = "glm-4-flash"  # 这个模型免费

# 豆包 Seed（字节跳动）
# client = OpenAI(api_key="你的key", base_url="https://ark.cn-beijing.volces.com/api/v3")
# model = "doubao-seed-1-6-250615"

# 通义千问（阿里）
# client = OpenAI(api_key="你的key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# model = "qwen-plus"

# DeepSeek
# client = OpenAI(api_key="你的key", base_url="https://api.deepseek.com")
# model = "deepseek-chat"

# ===== 调用方式都一样 =====
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "用一句话解释什么是 B+ 树索引"}]
)
print(response.choices[0].message.content)