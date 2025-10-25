import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_ms_marco")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_gpt2_ms_marco")
model.eval()

def generate_answer(question, max_new_tokens=50):
    prompt = f"Question: {question} Answer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("Answer:")[-1].strip()
    return answer

# Test examples
test_questions = [
    "What is the capital of France?",
    "How do you make coffee?"
]

for q in test_questions:
    ans = generate_answer(q)
    print(f"Q: {q}\nA: {ans}\n")

# Move model to GPU if available
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))