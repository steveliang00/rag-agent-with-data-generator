import os
import json
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from create_database import DocumentManager
from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from prompts_config import DATASET_CONFIGS, DEFAULT_CONFIG

# Load environment variables
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

class TrainingPrompt(BaseModel):
    """A structured representation of a training prompt."""
    user_content: str = Field(description="The training prompt")
    assistant_content: str = Field(description="The answer to the training prompt")


def initialize_generator(config_name: str = DEFAULT_CONFIG):
    """
    Initialize the LLM and prompt template based on configuration
    
    Args:
        config_name: Name of the configuration to use from DATASET_CONFIGS
        
    Returns:
        tuple: (llm, prompt, system_prompt)
    """
    if config_name not in DATASET_CONFIGS:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[config_name]
    
    # Initialize LLM with config settings
    # model_name = os.getenv("OPENROUTER_MODEL","openai/gpt-oss-20b:free")
    # llm = ChatOpenAI(
    #     api_key=openrouter_api_key,
    #     base_url=base_url,
    #     model=model_name,
    #     temperature=config["temperature"],
    #     timeout=30
    # )

    # MistralAI option
    model_name = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model=model_name,
        temperature=1,
        timeout=30
    )
    
    # Create prompt template
    parser = PydanticOutputParser(pydantic_object=TrainingPrompt)
    prompt = PromptTemplate(
        template=config["template"],
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    training_system_prompt = config["system_prompt"]
    return llm, prompt, parser, training_system_prompt



def generate_training_prompt(doc: Document, llm, prompt, parser):
    """
    Generates training prompts from one document chunk

    Args:
        doc: Document chunk
        llm: Language model instance
        prompt: Prompt template
        parser: Pydantic parser

    Returns:
        TrainingPrompt
    """    
    chain = prompt | llm | parser
    response = chain.invoke({"document_text": doc.page_content})
    return response

def format_prompt(content: TrainingPrompt, system_prompt: str):
    """
    Formats the training prompt into a JSON format
    Args:
        content: TrainingPrompt
        system_prompt: System prompt to use
    Returns:
        dict
    """
    user_content = content.user_content
    assistant_content = content.assistant_content
    
    output = {"messages":[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]}
    return output



def load_processed_chunks(chunks_file: str = "processed_chunks.pkl"):
    """
    Load pre-processed chunks from file
    
    Args:
        chunks_file: Path to the pickled chunks file
        
    Returns:
        List[Document]: The loaded chunks
    """
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"Chunks file '{chunks_file}' not found. "
            f"Please run 'python preprocess_chunks.py' first to generate the chunks file."
        )
    
    print(f"Loading pre-processed chunks from {chunks_file}...")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def write_single_jsonl_object(jsonl_file_path: str, obj: dict, mode: str = 'a'):
    """
    Writes a single JSON object to a JSONL file
    
    Args:
        jsonl_file_path: Path to the JSONL file
        obj: Dictionary to write
        mode: File mode ('w' for new file, 'a' for append)
    """
    with open(jsonl_file_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    # Configuration - change this to use different dataset types
    config_name = DEFAULT_CONFIG
    
    # Initialize generator with specified configuration
    print(f"Initializing generator with config: {config_name}")
    llm, prompt, parser, system_prompt = initialize_generator(config_name)
    
    # Load pre-processed chunks instead of processing PDF every time
    chunks = load_processed_chunks()
    
    # Define output file path
    output_file = f"training_data/training_data.jsonl"
    
    # Clear the file if it exists (optional - remove if you want to append to existing)
    # if os.path.exists(output_file):
    #     os.remove(output_file)
    
    print(f"Generating training data and writing to {output_file}...")
    
    # Process chunks and write incrementally
    for iter in range(1):
        for i, chunk in enumerate(chunks):
            # if i == 1:
            #     break  
            try:
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                
                # Generate training prompt
                training_prompt = generate_training_prompt(chunk, llm, prompt, parser)
                
                # Format the prompt
                formatted_data = format_prompt(training_prompt, system_prompt)
                
                # Write to JSONL file immediately
                write_single_jsonl_object(output_file, formatted_data, 'a')
                
                print(f"✓ Chunk {i+1} processed and written")
                
    
            except Exception as e:
                print(f"✗ Error processing chunk {i+1}: {e}")
        print(f"Iteration {iter+1} complete")
    
    print(f"Training data generation complete! Output written to {output_file}")