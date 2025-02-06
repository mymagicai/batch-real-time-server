from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import uvicorn
import torch
import os
import time

# Set environment variable before any CUDA initialization
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Pydantic models for request/response validation
class BatchRequest(BaseModel):
    prompts: List[str]

class BatchResponse(BaseModel):
    responses: List[str]
    total_time: float

# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# Initialize FastAPI app
app = FastAPI(title="vLLM Inference Server")

def format_chat_messages(messages: List[ChatMessage]) -> str:
    formatted_prompt = ""
    for msg in messages:
        if msg.role == "system":
            formatted_prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            formatted_prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            formatted_prompt += f"Assistant: {msg.content}\n"
    formatted_prompt += "Assistant: "
    return formatted_prompt

# Initialize vLLM with specific performance optimizations
def init_model(
    model_name: str = "xmadai/Llama-3.1-Nemotron-70B-Instruct-xMADai-INT4",
    tensor_parallel_size: int = 4
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This server requires GPU support.")
    
    device_count = torch.cuda.device_count()
    if device_count < tensor_parallel_size:
        raise RuntimeError(f"Requested {tensor_parallel_size} GPUs but only {device_count} are available")
    
    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size, 
        max_model_len=100000, 
        max_seq_len_to_capture=80000,  
        gpu_memory_utilization=0.95,  
        trust_remote_code=True,
        guided_decoding_backend="outlines",  
        enable_prefix_caching=True,  
        dtype="auto",
        swap_space=4,  
        max_num_batched_tokens=8192,  
        enforce_eager=True,
        device="cuda"  # Explicitly specify CUDA device
    )

# Initialize FastAPI application
def create_app():
    # Initialize model
    llm = init_model()
    
    # Fixed sampling parameters for all requests
    sampling_params = SamplingParams(
        max_tokens=2000,
        temperature=0,
        top_p=0.95,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    @app.post("/v1/generate_batch", response_model=BatchResponse)
    async def generate_batch(request: BatchRequest):
        try:
            start_time = time.time()
            
            # Batch inference with automatic batching by vLLM
            outputs = llm.generate(request.prompts, sampling_params)
            
            # Extract just the generated texts
            responses = [output.outputs[0].text for output in outputs]
            
            end_time = time.time()
            return BatchResponse(
                responses=responses,
                total_time=end_time - start_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        try:
            # Format chat messages into a prompt
            prompt = format_chat_messages(request.messages)
            
            # Create sampling parameters from request
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty
            )
            
            # Generate response
            outputs = llm.generate(prompt, sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Calculate token counts
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(outputs[0].outputs[0].token_ids)
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop"
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "model": "xmadai/Llama-3.1-Nemotron-70B-Instruct-xMADai-INT4",
            "gpu_count": torch.cuda.device_count()
        }
    
    return app

def main():
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # vLLM handles parallelization internally
        log_level="info"
    )

if __name__ == "__main__":
    main() 