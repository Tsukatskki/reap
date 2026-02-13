import sys
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, DataCollatorForLanguageModeling
from tqdm import tqdm

# Tooling imports
from reap.args import ModelArgs, DatasetArgs, FineTuneArgs
from reap.data import get_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def finetune_router(model, tokenizer, ds_args, steps=200, gradient_accumulation_steps=4):
    logger.info("🏥 Starting Router Rehabilitation (Fine-tuning)...")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # 1. Freeze non-router parameters
    model.train()
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        # Get input layer (usually get_input_embeddings()) and register hook
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "router" in name or "gate" in name: # Handle model-specific naming
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    logger.info(f"Trainable params: {trainable_params} / {all_params} ({(trainable_params/all_params)*100:.4f}%)")

    # 2. Prepare optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # 3. Prepare data
    # Ensure at least 200 samples (with BS=1)
    req_samples = max(steps * 2, 800) 
    logger.info(f"Loading {req_samples} samples for tuning...")
    
    dataset = get_dataset(
        ds_args.dataset_name, 
        tokenizer, 
        split="train", 
        num_samples=req_samples 
    )
    
    # 4. DataLoader (critical fix: batch size = 1)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,  # <--- Prevent OOM
        shuffle=True, 
        collate_fn=data_collator
    )
    
    # 5. Training loop
    progress_bar = tqdm(range(steps), desc="Router Tuning")
    iter_loader = iter(dataloader)
    total_loss = 0
    device = model.device
    logger.info(f"🎮 Training on device: {device}")

    for step in range(steps):
        # Gradient accumulation loop
        accumulated_loss = 0
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                batch = next(iter_loader)
                
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model(**inputs)

            # loss
            step_loss = outputs.loss
            loss_value = step_loss.item() 
            loss = step_loss / gradient_accumulation_steps
            
            # Backward (accumulate gradients)
            loss.backward()
            accumulated_loss += loss_value
            
            # Free GPU memory
            del inputs, outputs, loss
        
        # Update parameters only after accumulation
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += accumulated_loss
        
        if step % 10 == 0:
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{accumulated_loss:.4f}", "avg": f"{avg_loss:.4f}"})
        
        progress_bar.update(1)

    progress_bar.close()
    logger.info("Router fine-tuning completed.")
    return model

def main():
    # Load the pruned model and fine-tune the router
    parser = HfArgumentParser((ModelArgs, DatasetArgs, FineTuneArgs))
    model_args, ds_args, finetune_args = parser.parse_args_into_dataclasses()

    # Show GPU info
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    logger.info(f"🎮 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    logger.info(f"🎮 Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 Note: Physical GPUs {cuda_visible} are mapped to cuda:0, cuda:1, ... in PyTorch")
        for i in range(torch.cuda.device_count()):
            logger.info(f"🎮 cuda:{i} -> {torch.cuda.get_device_name(i)}")
    
    # 1. Load the pruned model (local path)
    logger.info(f"Loading PRUNED model from: {model_args.model_name}")
    
    # The config must tolerate a smaller num_experts
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, # Points to prune_model.py output directory
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
    )
    
    # Show model device placement
    if hasattr(model, 'hf_device_map'):
        logger.info(f"🎮 Model device map: {model.hf_device_map}")
    else:
        logger.info(f"🎮 Model device: {model.device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    
    # Fallback: if tokenizer has no chat_template, add one
    if tokenizer.chat_template is None:
         tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # 2. Run fine-tuning
    logger.info(f"Fine-tuning router for {finetune_args.router_finetune_steps} steps...")
    model = finetune_router(
        model, 
        tokenizer, 
        ds_args, 
        steps=finetune_args.router_finetune_steps,
        gradient_accumulation_steps=finetune_args.gradient_accumulation_steps
    )

    # 3. Save model (overwrite original)
    logger.info(f"Saving fine-tuned model back to {model_args.model_name}")
    model.save_pretrained(model_args.model_name)
    tokenizer.save_pretrained(model_args.model_name)
    logger.info("✅ Router fine-tuning completed and model saved!")

if __name__ == "__main__":
    main()