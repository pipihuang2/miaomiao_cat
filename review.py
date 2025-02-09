import os
import requests
import openai
# DeepSeek API 客户端初始化
class OpenAI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def generate_review(self, code):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": f"请审查以下代码并提出改进建议:\n\n{code}",
            "max_tokens": 1500,
            "temperature": 0.7
        }
        response = requests.post(
            f"{self.base_url}/v1/completions", json=payload, headers=headers
        )
        return response.json().get('choices', [{}])[0].get('text', '').strip()

# 使用 DeepSeek API 客户端
client = OpenAI(
    api_key="sk-ff0b0846981844eba62e7920994ba65a",  # 替换为你的 API 密钥
    base_url="https://api.deepseek.com"  # DeepSeek 的 API 基础地址
)

# 获取 PR 代码（假设你已获取到代码，可能通过 GitHub API 获取 PR 差异）
def get_pr_code():
    with open('example_code.py', 'r') as file:  # 假设你将 PR 代码保存为文件
        return file.read()

# 使用 DeepSeek API 进行代码审查
def review_code(code):
    return client.generate_review(code)

# 将审查结果发送到飞书
def send_to_feishu(review_result):
    feishu_webhook_url = os.getenv("FEISHU_WEBHOOK_URL")
    headers = {"Content-Type": "application/json"}
    payload = {
        "msg_type": "text",
        "content": {
            "text": f"以下是代码审查的建议：\n\n{review_result}"
        }
    }
    response = requests.post(feishu_webhook_url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Successfully sent to Feishu")
    else:
        print(f"Failed to send message: {response.status_code}")

if __name__ == "__main__":
    pr_code = get_pr_code()
    review_result = review_code(pr_code)
    send_to_feishu(review_result)
