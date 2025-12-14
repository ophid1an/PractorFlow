

from llm import LLMConfig, create_runner
from llm.base.session import Session

config = LLMConfig(model_name="qwen-instruct")
session = Session("internal-demo")

ai = create_runner(config, session=session)
ai.load_document("company_docs.pdf")

response = ai.generate(
    prompt="Summarize our Q4 roadmap",
    use_context=True
)
