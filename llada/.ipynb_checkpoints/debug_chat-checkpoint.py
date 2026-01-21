import torch
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM

# ============ 在这里随意修改参数 ============
gen_length = 128
steps = 128
block_size = 32
use_cache = True
if_cache_position = False
threshold = 0.9
# ==========================================

device = 'cuda'
model = LLaDAModelLM.from_pretrained(
    'GSAI-ML/LLaDA-8B-Instruct', 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    'GSAI-ML/LLaDA-8B-Instruct', 
    trust_remote_code=True
)

print('*' * 66)
print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
print(f'**  Use Cache: {use_cache}  |  Cache Position: {if_cache_position}  **')
print(f'**  Threshold: {threshold}  |  Block Size: {block_size}  **')
print('*' * 66)

conversation_num = 0
prompt = None

while True:
    user_input = input("Enter your question: ")
    
    m = [{"role": "user", "content": user_input}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(user_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    if conversation_num == 0:
        prompt = input_ids
    else:
        prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
    
    # 选择生成方法
    if use_cache:
        if if_cache_position:
            out, nfe = generate_with_dual_cache(
                model, prompt, steps=steps, gen_length=gen_length, 
                block_length=block_size, temperature=0., 
                remasking='low_confidence', threshold=threshold
            )
        else:
            out, nfe = generate_with_prefix_cache(
                model, prompt, steps=steps, gen_length=gen_length, 
                block_length=block_size, temperature=0., 
                remasking='low_confidence', threshold=threshold
            )
    else:
        out, nfe = generate(
            model, prompt, steps=steps, gen_length=gen_length, 
            block_length=block_size, temperature=0., 
            remasking='low_confidence', threshold=threshold
        )
    
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print(f"\nBot's reply: {answer}")
    print(f"Number of forward passes: {nfe}")
    
    prompt = out[out != 126081].unsqueeze(0)
    conversation_num += 1
    print('-' * 70)