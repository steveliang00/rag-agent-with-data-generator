# RAG Agent & Training Dataset Generator
- A simple RAG chatbot to answer questions about your pdf documents
- A script to generate training data for fine-tuning from your documents

Streamlit app of chatbot here:
https://the-pdf-rag-chatbot.streamlit.app/
## Instructions
### RAG Agent
1. Add pdf documents to `data` folder
2. run `main.py` to chat with your documents

### Training Dataset Generator
1. Create a file named `prompts_config.py` and add the following variables
```python
TRAINING_TEMPLATE = """
This is your system prompt for the generator
"""

TRAINING_SYSTEM_PROMPT = """
This is the system prompt you want to append to each training datapoint
"""

# Can add multiple configurations
DATASET_CONFIGS = {
    "default": {
    "template": TRAINING_TEMPLATE,
    "system_prompt": TRAINING_SYSTEM_PROMPT,
    "model": "openai/gpt-oss-20b:free",
    "temperature": 0.9
    }
}
DEFAULT_CONFIG = "default"
```
2. Configure your llm in the `initialize_generator` function
3. run `generate_dataset.py`


