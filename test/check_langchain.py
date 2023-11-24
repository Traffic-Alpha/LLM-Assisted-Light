'''
@Author: WANG Maonan
@Date: 2023-10-15 23:11:49
@Description: Use this file to test whether your gpt works
@LastEditTime: 2023-11-24 22:41:33
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))


from utils.readConfig import read_config
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

custom_style = """American English \
in a calm and respectful tone, at last translate to Simplified Chinese
"""


if __name__ == '__main__':
    config = read_config()
    openai_proxy = config['OPENAI_PROXY']
    openai_api_key = config['OPENAI_API_KEY']
    openai_api_base = config['OPENAI_API_BASE']
    openai_model = config['OPENAI_API_MODEL']

    # 模型初始化
    chat = ChatOpenAI(
        model=openai_model, 
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy,
        openai_api_base=openai_api_base,
        temperature=0.0,
    )

    # 创建模板 (这个模板可以传入不同的参数进行重复使用)
    template_string = """Translate the text \
    that is delimited by triple backticks 
    into a style that is {style}.
    text: ```{text}```
    """
    prompt_templete = ChatPromptTemplate.from_template(template_string)
    # 传入不同的参数
    custom_message = prompt_templete.format_messages(
        style = custom_style,
        text = customer_email
    )
    
    # 将 prompt 传入 chat
    print(f'传入的信息:\n{custom_message}')
    custom_response = chat(custom_message)
    print(f'回答的结果:\n{custom_response}')