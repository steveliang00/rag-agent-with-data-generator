import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from create_database import DocumentManager
from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

class TrainingPrompt(BaseModel):
    """A structured representation of a training prompt."""
    user_content: str = Field(description="The training prompt")
    assistant_content: str = Field(description="The answer to the training prompt")



llm = ChatOpenAI(
            api_key=openrouter_api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.9,
            timeout=30
        )

template = """
You are tasked with creating a training example for fine-tuning an LLM to understand and apply medical benefit grammar rules from SBC (Summary of Benefits and Coverage) documents. 
Based on the attached snippet from the Medical Benefit Grammar Guide, pick one of the following example training formats to generate from:
**TRAINING EXAMPLE FORMATS:**

1. **Direct Fact Recall**: Simple question-answer pairs about specific field mappings, definitions, and rules
   Format: Question → Direct answer

2. **Rule Application**: Questions requiring explanation of how to apply specific grammar rules or conventions
   Format: Question → Detailed step-by-step explanation

3. **Scenario-Based**: Real-world situations requiring decision-making based on the grammar rules
   Format: Scenario description → Correct action/choice with reasoning

**Direct Fact Recall Examples:**
{{"messages": [{{"role": "user", "content": "What is the definition of Actuarial Value?"}}, {{"role": "assistant", "content": "The percentage of total average costs for covered benefits that a plan will pay for. For example, if a plan has an actuarial value of 70%, on average, you would be responsible for 30% of the costs of all covered benefits."}}]}}

**Rule Application Examples:**
{{"messages": [{{"role": "user", "content": "How do I handle varying cost share per visit?"}}, {{"role": "assistant", "content": "When benefits have different cost shares depending on the number of visits, use the format: 'first # visit(s) [costshare] then [costshare]'. For example: 'first 3 visit(s) $75 then $75 after deductible'. If deductible only applies to the first cost share, place 'after deductible' following that amount. If it only applies to the second cost share, place it at the end of the benefit string. If it applies to both, include 'after deductible' after both benefit strings."}}]}}
{{"messages": [{{"role": "user", "content": "How should I capture Emergency Room benefits when both copay and coinsurance values are listed?"}}, {{"role": "assistant", "content": "When both copay and coinsurance values are listed for Emergency Room, capture it as '$x plus y%'. For example, if there's a $300 copay plus 20% coinsurance, capture it as '$300 plus 20%'. This format is specifically common for Emergency Room benefits when multiple cost-sharing mechanisms apply to the same service."}}]}}

**Scenario-Based Examples:**
{{"messages": [{{"role": "user", "content": "I have an SBC that shows different benefits for Hospital vs. Freestanding Facility for Diagnostic Test. Which should I capture?"}}, {{"role": "assistant", "content": "Capture the values for Freestanding Facility. According to the grammar rules, when benefits are specified for Freestanding vs. Hospital settings for Diagnostic Test, you should capture the Freestanding Facility values as they represent the non-hospital, outpatient testing environment that this field is designed to capture."}}]}}
{{"messages": [{{"role": "user", "content": "An SBC shows 'prophylaxis' coverage for adults. How should I handle this for the Adult Dental field?"}}, {{"role": "assistant", "content": "Mark the Adult Dental field as True. The grammar guide specifically states that benefits listed as 'Prophylaxis' should be captured under Adult Dental, as this refers to routine/basic adult dental care coverage. Since prophylaxis is mentioned, this indicates there is coverage for routine adult dental services."}}]}}

Document Text:
{document_text}

Generate ONE training example based on the document text above.

Please format the output as a JSON object with the following structure:
{format_instructions}
"""

training_system_prompt = """
You are an AI assistant specialized in extracting and structuring medical benefit information from SBC documents according to Ideon's Medical Benefit Grammar Guide. 
Your role is to accurately interpret SBC data, apply the correct grammar rules and conventions, and provide precise field mappings and benefit captures.
"""

parser = PydanticOutputParser(pydantic_object=TrainingPrompt)

prompt = PromptTemplate(
    template=template,
    input_variables=["document_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

def generate_training_prompt(doc: Document):
    """
    Generates training prompts from one document chunk

    Args:
        chunks: Document

    Returns:
        TrainingPrompt
    """    
    chain = prompt | llm | parser
    response = chain.invoke({"document_text": doc.page_content})
    return response

def format_prompt(content: TrainingPrompt):
    """
    Formats the training prompt into a JSON format
    Args:
        content: TrainingPrompt
    Returns:
        str
    """
    user_content = content.user_content
    assistant_content = content.assistant_content
    
    output = {"messages":[
        {"role": "system", "content": training_system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]}
    return output



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
    document_manager = DocumentManager()
    docs = document_manager.load_docs()
    chunks = document_manager.split_text(docs)
    
    # Define output file path
    output_file = "training_data.jsonl"
    
    # Clear the file if it exists (optional - remove if you want to append to existing)
    # if os.path.exists(output_file):
    #     os.remove(output_file)
    
    print(f"Generating training data and writing to {output_file}...")
    
    # Process chunks and write incrementally
    for i, chunk in enumerate(chunks):
        # if i == 3:
        #     break  
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Generate training prompt
            training_prompt = generate_training_prompt(chunk)
            
            # Format the prompt
            formatted_data = format_prompt(training_prompt)
            
            # Write to JSONL file immediately
            write_single_jsonl_object(output_file, formatted_data, 'a')
            
            print(f"✓ Chunk {i+1} processed and written")
            
 
        except Exception as e:
            print(f"✗ Error processing chunk {i+1}: {e}")
            break
    
    print(f"Training data generation complete! Output written to {output_file}")