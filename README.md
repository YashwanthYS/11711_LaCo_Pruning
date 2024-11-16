# LaCo Pruning

This repository contains the code to reproduce the results mentioned in the [LaCo paper](https://arxiv.org/pdf/2402.11187).

## Components
- **Prune**: Loads the model from HuggingFace, then prunes the model weights, and pushes the model architecture and weights to HuggingFace for easier inference and analysis.
- **Evaluation**: Loads the pruned model's weights from HuggingFace, and uses [OpenCompass](https://github.com/open-compass/opencompass) for evaluation on tasks such as Understanding, Reasoning, and Language.
- **Plots**: Contains the code to reproduce the plots used by author's in the original paper.


## Pruning
- Prunes layers of the LLaMA-2 model using a similarity-based merging approach.
- Saves the pruned model and tokenizer for further use.
- Publishes the pruned model to the Hugging Face Hub.

  Add image here TO-DO

For pruning, we have adapted the code for our use based on the author's official code implementation [LaCo](https://github.com/yangyifei729/laco).

### Steps to Reproduce
You can refer to the [notebook](https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/prune/Llama2_7b_LaCo.ipynb) which prunes the LLaMa2-7B model based on the hyperparameters used in the paper.
1. **Log in to Hugging Face Hub**:
   Use your Hugging Face token to log in:
   ```python
   from huggingface_hub import login
   HF_TOKEN = "<your_huggingface_token>"
   login(HF_TOKEN)
   ```
2. **Load the Pre-trained Model**:
   Load the model and tokenizer from the Hugging Face Hub.
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   llama_path = 'meta-llama/Llama-2-7b-hf'
   llama_model = AutoModelForCausalLM.from_pretrained(llama_path, trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
   ```

3. **Set Pruning Parameters**:
   Initialize parameters such as interval, merge layers, and similarity threshold:
   ```python
   INTERVAL = 2
   MERGE_LAYERS = 4
   HIGHEST_LAY = 32
   LOWEST_LAY = 1
   THRESHOLD = 0.65
   ```

4. **Define the Layer Merging Function**:
   Use the `merge_layers_return_model` function to merge layers.

5. **Iteratively Prune Layers**:
   Reduce the model's layers while maintaining a similarity above the threshold.

6. **Save the Pruned Model**:
   Save the compressed model and tokenizer locally:
   ```python
   llama_copy_to_compress.save_pretrained("output_path/pruned_model")
   tokenizer.save_pretrained("output_path/pruned_model")
   ```

7. **Upload the Model** (optional):
   Publish the pruned model to Hugging Face Hub:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo(repo_id="your_repo_name", exist_ok=True)
   api.upload_folder(
       folder_path="output_path/pruned_model",
       repo_id="your_repo_name",
       repo_type="model",
   )
   ```
