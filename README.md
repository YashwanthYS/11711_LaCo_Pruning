# LaCo Pruning

This repository contains the code to reproduce the results mentioned in the [LaCo paper](https://arxiv.org/pdf/2402.11187). Developed as a part of the Assignment 3 for the [CMU ANLP Course 11-711](https://www.phontron.com/class/anlp-fall2024/)

## Components
- **Prune**: Loads the model from HuggingFace, then prunes the model weights, and pushes the model architecture and weights to HuggingFace for easier inference and analysis. [Link](https://github.com/YashwanthYS/11711_LaCo_Pruning/tree/main/prune)
- **Evaluation**: Loads the pruned model's weights from HuggingFace, and uses [OpenCompass](https://github.com/open-compass/opencompass) for evaluation on tasks such as Understanding, Reasoning, and Language. [Link](https://github.com/YashwanthYS/11711_LaCo_Pruning/tree/main/experiments)
- **Plots**: Contains the code to reproduce the plots used by author's in the original paper. [Link](https://github.com/YashwanthYS/11711_LaCo_Pruning/tree/main/plots)


## Pruning
- Prunes layers of the LLaMA-2 model using a similarity-based merging approach.
- Saves the pruned model and tokenizer for further use.
- Publishes the pruned model to the Hugging Face Hub.

  <img src="https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/imgs/LaCo_prune.png" alt="Pruning Pipeline" width="700"/>


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

4. **Iteratively Prune Layers**:
   Reduce the model's layers while maintaining a similarity above the threshold using the code.

5. **Save the Pruned Model**:
   Save the compressed model and tokenizer locally:
   ```python
   llama_copy_to_compress.save_pretrained("output_path/pruned_model")
   tokenizer.save_pretrained("output_path/pruned_model")
   ```

6. **Upload the Model** (optional):
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


## Inference and Evaluation
- Uses OpenCompass to load the relevant datasets for evaluation.
- Evaluates the performance of pruned model on selected datasets.
- Writes the output to a markdown and csv file.

  <img src="https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/imgs/LaCo_infer.png" alt="Inference Pipeline" width="700"/>


### Steps to Reproduce
You can refer to the [notebook](https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/experiments/llama-7B/OpenCompassEval_llama2_7b_Benchmarks.ipynb) which runs the evaluation on selected datasets using the pruned LLaMa2-7B model.
1. **Install OpenCompass**:
   
 ```python
  !git clone https://github.com/open-compass/opencompass.git
  ```
  
  ```python
    %cd opencompass
  ```
  
  ```python
    !pip install -e .
  ```
2. **Run Evaluation Script**:
 ```python
  !python run.py \
      --datasets hellaswag_ppl \
      --hf-type base \
      --hf-path enter_HF_path \
      --tokenizer-path enter_HF_path \
      --model-kwargs device_map='auto' \
      --max-seq-len 1024 \
      --max-out-len 100 \
      --min-out-len 100 \
      --batch-size 8 \
      --hf-num-gpus 1
  ```
3. **Zip results and download the prediction files with metrics**:
  ```python
from google.colab import files

!zip -r /path/to/output.zip /content/opencompass/outputs
files.download('/path/to/output.zip')
```

## Plots
  This component contains the part to reproduce the plots authors have implemented in the paper. It contains separate notebooks containing the code and detailed steps to generate plots related to
  - [RDSC Layer Merge Similarity](https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/plots/plot_RDSC_cosine.ipynb)
  - [Cosine Similarity between outputs of successive layers](https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/plots/plot_cosine_similarities.ipynb)

### Sample Plots
1. **Cosine Similarity between adjacent layer outputs**:
   
   <img src="https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/imgs/cosine_similarities_7B_models.jpg" alt="Adj layer similarities" width="550"/>

2. **RDSC Layer merge cosine similarity**:
   
   <img src="https://github.com/YashwanthYS/11711_LaCo_Pruning/blob/main/imgs/rdsc_7b.jpg" alt="RDSC cosine similarities" width="550"/>
