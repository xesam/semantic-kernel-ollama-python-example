import asyncio
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.ollama.ollama_prompt_execution_settings import (
    OllamaChatPromptExecutionSettings,
)

kernel = Kernel()

service_id = "ollama"
kernel.add_service(
    OllamaChatCompletion(
        service_id=service_id,
        host="http://localhost:11434",
        ai_model_id="llama3.1",
    )
)
execution_settings = OllamaChatPromptExecutionSettings()

prompt = """
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words."""

prompt_template_config = PromptTemplateConfig(
    template=prompt, name="tldr", template_format="semantic-kernel"
)

function = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt_template_config=prompt_template_config,
    prompt_execution_settings=execution_settings,
)


# Run your prompt
# Note: functions are run asynchronously
async def main():
    result = await kernel.invoke(function)
    print(result)  # => Robots must not harm humans.


if __name__ == "__main__":
    asyncio.run(main())
