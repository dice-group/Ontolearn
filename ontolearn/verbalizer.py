# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

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