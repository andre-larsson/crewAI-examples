from langchain_openai import ChatOpenAI

# Outline writer

# uncensored, but only ollama-compatible version of longwriter I found
longwriter_llm = "ollama/hf.co/mradermacher/DarkIdol-LongWriter-v13-8B-Uncensored-1048k-GGUF"

outline_llm = ChatOpenAI(model="ollama/mistral-small", base_url="http://localhost:11434/slv1", max_tokens=2000, temperature=0.7, max_retries=3)

# tool calling required
outline_researcher_llm = ChatOpenAI(model="ollama/llama3.1", base_url="http://localhost:11434/slv1")


# Chapter writer

chapter_llm = ChatOpenAI(model=longwriter_llm, base_url="http://localhost:11434/slv1", max_tokens=10000, temperature=0.2, max_retries=3)

# tool calling required
chapter_researcher_llm = ChatOpenAI(model="ollama/llama3.1", base_url="http://localhost:11434/slv1")