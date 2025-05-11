import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def locate_concept_dims(model_name: str,
                        concept: str,
                        device: str = 'cuda',
                        top_n: int = 20):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    E_in = model.get_input_embeddings().weight.detach()  
    token_ids = tokenizer(concept, add_special_tokens=False).input_ids
    with torch.no_grad():
        concept_emb = E_in[token_ids].mean(dim=0)  
        concept_emb = concept_emb / concept_emb.norm()  
    
    scores = [] 
    
    for layer_idx, block in enumerate(model.model.layers):
        W = block.mlp.up_proj.weight.detach()  # shape (d_i, d)
        v_normed = W / (W.norm(dim=1, keepdim=True) + 1e-6)
        with torch.no_grad():
            sims = v_normed @ concept_emb      
        
        topk = torch.topk(sims, k=min(top_n, sims.shape[0]))
        for score, dim_idx in zip(topk.values.tolist(), topk.indices.tolist()):
            scores.append((score, layer_idx, dim_idx))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    print(f"Top {top_n} concept dims for “{concept}” in {model_name}:\n")
    for score, layer_idx, dim_idx in scores[:top_n]:
        print(f"  Layer {layer_idx:2d}  Dim {dim_idx:4d}   Score {score:.4f}")
if __name__ == "__main__":
    locate_concept_dims("/home/tlmsq/rlrover/checkpoint-1868", "Gibbon", device="cuda", top_n=10)