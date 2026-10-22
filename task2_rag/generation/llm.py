from openai import OpenAI


class QwenLLM:


    def __init__(self):

        self.client = OpenAI(

            base_url="http://127.0.0.1:8000/v1",

            api_key="EMPTY"

        )

        self.model = "qwen-local"



    def chat(
            self,
            question: str,
            contexts: list[str]
    ):
        context_text = "\n\n".join(contexts)

        # 控制输入长度，适配当前512 token模型
        context_text = context_text[:1200]


        prompt = f"""
你是一个专业保险助手。

请根据下面提供的保险条款回答用户问题。
如果资料中没有答案，请明确说明无法从条款中找到。

保险条款：

{context_text}


用户问题：

{question}


请给出准确、简洁的回答。
"""


        response = self.client.chat.completions.create(

            model=self.model,

            messages=[

                {
                    "role": "system",
                    "content": "你是一个保险条款问答助手。"
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0,
            max_tokens=80,

        )


        return response.choices[0].message.content
