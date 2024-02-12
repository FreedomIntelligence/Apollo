# Multilingual Medical Basic (EN/ZH/ES/FR/AR/HI): Model, Dataset, Benchmark, Code




<center>

![Python 3.10](https://img.shields.io/badge/Python-3.10-lightblue) ![Pytorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-lightblue) ![transformers](https://img.shields.io/badge/transformers-4.34.0.dev0%2B-lightblue) ![accelerate](https://img.shields.io/badge/accelerate-0.22-lightblue)
</center>


![Medbase](assets/Medbase.png)

<p align="center">
   📃 <a href="" target="_blank">Paper</a> • 🌐 <a href="" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_data" target="_blank">Medbase_data</a> • 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_eval" target="_blank">Medbase_eval</a> 
   <br>  <a href="./README_zh.md">   中文</a> | <a href="./README.md"> English
</p>

     

## 🌈 Update

**Waiting for update, coming soon**

* **[2024.02.12]** <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_data" target="_blank">Medbase_data</a> and  <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_eval" target="_blank">Medbase_eval</a>  is published！🎉
* **[2024.01.23]** Medbase repo is published！🎉


## Results
   🤗 <a href="" target="_blank">Medbase_0.5B</a> • 🤗 <a href="" target="_blank">Medbase_1.8B</a> • 🤗 <a href="" target="_blank">Medbase_1.8B * 4</a>  • 🤗 <a href="" target="_blank">Medbase_1.8B * 8</a> • 🤗 <a href="" target="_blank">Medbase_6B</a> 
   <details><summary>Click to expand</summary>
   
   
   **Results and Models are coming soon !**
      
   
   
   </details>


## Dataset & Evaluation

- Dataset
  🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_data" target="_blank">Medbase_data

- Evaluation
  🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_eval" target="_blank">Medbase_eval</a> 


## Results reproduction
   <details><summary>Click to expand</summary>
   
   1. Prepare Train/Test Data
      - [Back Translation using LLMs](https://github.com/FreedomIntelligence/Medbase/tree/main/src/process/openai_rewrite): Run Bash File
      - [Prepare Training tokens for LLMs](https://github.com/FreedomIntelligence/Medbase/tree/main/src/process/prepare): Run Bash File
   2. [Train your model](https://github.com/FreedomIntelligence/Medbase/tree/main/src/sft): Run Bash file
   3. [Evaluation](https://github.com/FreedomIntelligence/Medbase/tree/main/src/evaluate): Run Bash file
   
   </details>



##  Acknowledgment

This Repo is highly dependent on [HuatuoGPT-II](https://github.com/FreedomIntelligence/HuatuoGPT-II)

##  Citation
Please use the following citation if you intend to use our dataset for training or evaluation:

```
@misc{medbase,
  title={MedBase, Exploring the boundaries of open source LLM medical capabilities},
  author={Xidong Wang, Junyin Chen, Nuo Chen, Yidong Wang, Zhiyi Zhang, Benyou Wang},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FreedomIntelligence/Medbase}},
}
```
