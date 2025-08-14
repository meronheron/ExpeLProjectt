from typing import Callable, List
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import openai


class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool):
        self.model_name = llm_name
        if long_ver:
            llm_name = 'gpt-3.5-turbo-16k'
        self.llm = ChatOpenAI(
            model=llm_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        # Debug print: show messages being sent
        print("\n[DEBUG] Messages being sent to LLM:")
        for msg in messages:
            print(f"- {msg.content}")

        kwargs = {}
        if stop:
            kwargs['stop'] = stop

        for i in range(6):
            try:
                output = self.llm(messages, **kwargs).content.strip('\n').strip()
                break
            except openai.error.RateLimitError:
                print(f'\nRetrying {i}...')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output


def LLM_CLS(llm_name: str, openai_api_key: str, long_ver: bool) -> Callable:
    if 'gpt' in llm_name:
        wrapper = GPTWrapper(llm_name, openai_api_key, long_ver)
        # Create a callable lambda but attach the wrapper to it
        def llm_callable(messages, stop=[], replace_newline=True):
            return wrapper(messages, stop, replace_newline)
        llm_callable.wrapper = wrapper  # attach original wrapper
        return llm_callable
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}")
