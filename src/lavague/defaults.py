from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class DefaultEmbedder(HuggingFaceEmbedding):
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cuda"):
        super().__init__(model_name, device)


from llama_index.llms.openai import OpenAI
import os


class DefaultLLM(OpenAI):
    def __init__(self):
        max_new_tokens = 512
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        else:
            super().__init__(api_key=api_key, max_tokens=max_new_tokens)




def default_get_driver() -> PlaywrightDriver:
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--window-size=1600,900"])
    page = browser.new_page()
    return PlaywrightDriver(page)
