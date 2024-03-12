import requests


class LLMVerbalizer:
    def __init__(self, model: str = "mixtral:8x7b",
                 url: str = "http://diceemb.cs.upb.de:8000/api/generate"):
        self.model = model
        self.url = url

    def __call__(self, text: str):
        """
        :param text: String representation of an OWL Class Expression
        """
        prompt = "You are a Description Logic expert. You are particularly good at explaining a complex Description Logic concepts in few sentences."
        prompt += f"Explain {text} in two sentences."
        response = requests.get(url=self.url,
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"key": 84, "model": self.model, "prompt": prompt})
        return response.json()["response"]
