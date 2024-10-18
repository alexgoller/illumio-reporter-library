from .base_model import BaseAIModel
import requests
import json

class OllamaModel(BaseAIModel):
    def __init__(self, model: str = "mistral", api_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = f"{api_url}/api/generate"

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "prompt": prompt
        }
        try:
            response = requests.post(self.api_url, json=data, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line)
                        if 'response' in json_line:
                            full_response += json_line['response']
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line as JSON: {line}")

            return full_response
        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to Ollama API: {str(e)}"
            print(error_message)
            return error_message
