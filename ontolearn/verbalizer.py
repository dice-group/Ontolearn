import requests


class LLMVerbalizer:
    def __init__(self, model: str = "mixtral:8x7b",
                 url: str = "http://tentris-ml.cs.upb.de:8000/api/generate"):
        self.model = model
        self.url = url

    def __call__(self, text: str):
        """
        :param text: String representation of an OWL Class Expression
        """
        prompt = f"<s> [INST] You are an expert in description logics. You are particularly good at explaining complex concepts with few sentences. [/INST] Model answer</s> [INST] Verbalize {text} in natural language with 1 sentence. Provide no explanations or write no notes.[/INST]"
        print("Waiting for the verbalization..")
        try:
            response = requests.get(url=self.url,
                                    headers={"accept": "application/json", "Content-Type": "application/json"},
                                    json={"model": self.model, "prompt": prompt}, timeout=30)
            if response.ok:
                return response.json()["response"]
            else:
                return f"No verbalization due to the HTTP connection\t{response.text}"
        except:
            return f"No verbalization. Error at HTTP connection"