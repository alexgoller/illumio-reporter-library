from .base_model import BaseAIModel
from anthropic import Anthropic

class AnthropicModel(BaseAIModel):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
