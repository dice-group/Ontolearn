import requests

from openai import OpenAI
from typing import Optional, Union
from ontolearn.utils.static_funcs import assert_class_expression_type

class LLMVerbalizer:
    def __init__(self, model: str = "mixtral:8x7b",
                 url: str = "http://tentris-ml.cs.upb.de:8000/api/generate", api_key: Optional[str] = ""):
        self.model = model
        self.url = url
        self.api_key = api_key
        self.client = OpenAI(base_url=self.url, api_key=self.api_key)

    def __call__(self, text: str, api_endpoint: Optional[bool] = True):
        """
        :param text: String representation of an OWL Class Expression
        """
        assert isinstance(text, str) or assert_class_expression_type(text), "Input must be a string or either of the family OWL class expression"

        prompt = f"<s> [INST] You are an expert in description logics. You are particularly good at explaining complex concepts with few sentences. [/INST] Model answer</s> [INST] Verbalize {text} in natural language with 1 sentence. Provide no explanations or write no notes.[/INST]"
        print("Waiting for the verbalization..")
        try:
            if api_endpoint:
                response = requests.get(url=self.url,
                                        headers={"accept": "application/json", "Content-Type": "application/json"},
                                        json={"model": self.model, "prompt": prompt}, timeout=30)
                if response.ok:
                    return response.json()["response"]
                else:
                    return f"No verbalization due to the HTTP connection\t{response.text}"
            else:
                assert (self.api_key != ""), "Client API key is not empty"

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}]
                )

                return (response.choices[0].message.content)

        except:
            return f"No verbalization. Error at HTTP connection"
        
def verbalize_learner_prediction(prediction: Union[str, object] = None):
    if prediction is None:
        raise ValueError("Learner prediction cannot be None")
    
    verbalizer = LLMVerbalizer(url='http://tentris-ml.cs.upb.de:8501/v1', model='tentris', api_key='') #ENTER YOUR API KEY
    predicitions = [verbalizer(text=prediction, api_endpoint=False) for _ in range(3)]
    print(predicitions)


