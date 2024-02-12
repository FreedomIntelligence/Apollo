# Multilingual Medical Basic Model (En/ZH/ES/FR/AR/HI)




<center>

![Python 3.10](https://img.shields.io/badge/Python-3.10-lightblue) ![Pytorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-lightblue) ![transformers](https://img.shields.io/badge/transformers-4.34.0.dev0%2B-lightblue) ![accelerate](https://img.shields.io/badge/accelerate-0.22-lightblue)
</center>


![Medbase](assets/Medbase.png)

<p align="center">
   📃 <a href="" target="_blank">Paper</a> • 🌐 <a href="" target="_blank">Website</a> • 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/Medbase_data" target="_blank">Medbase_data</a>  
   <br> 🤗 <a href="" target="_blank">Medbase_0.5B</a> • 🤗 <a href="" target="_blank">Medbase_1.8B</a> • 🤗 <a href="" target="_blank">Medbase_1.8B*4</a>  • 🤗 <a href="" target="_blank">Medbase_1.8B*8</a> • 🤗 <a href="" target="_blank">Medbase_6B</a> 
   <br>  <a href="./README_zh.md">   中文</a> | <a href="./README.md"> English
</p>

     

## 🌈 Update

**Waiting for update, coming soon**

* **[2024.01.23]** Medbase repo is published！🎉


## Results
   <details><summary>Click to expand</summary>
   
   
   **More Results and Models are coming soon !**
      
   | Model          | MedQA-USMLE | MedMCQA | PubMedQA | MMLU-Medical | MedQA-MCMLE | CMB-single | CMMLU-Medical | CExam |
   | -------------- | ----------- | ------- | -------- | ------------ | ----------- | ---------- | ------------- | ----- |
   | Qwen-1.8B-chat | 27.42       | 29.18   | 34.90    | 37.47        | 44.25       | 31.40      | 37.28         | 30.65 |
   | Qwen-1.8B      | 26.71       | 30.34   | 49.30    | 41.10        | 44.63       | 33.15      | 37.96         | 34.50 |
   | **Medbase-1.8B**   | **45.01**       | **48.00**   | **53.00**    | **53.39**        | **76.15**       | **56.15**      | **57.46**         | **61.50** |
   | Llama2-7B      | 25.84       | 32.76   | 43.20    | 33.51        | 25.10       | 20.75      | 23.78         | 20.65 |
   | Huatuo2-7B     | 41.13       | 41.87   |          | 51.44        |             |            | 59.08         | 65.81 |
   | Mistral-7B     | 41.10       | 40.20   | 17.80    | 55.80        |             |            |               |       |
   | PMC-Llama-7B   | 49.20       | 57.60   | 59.20    | 59.70        |             |            |               |       |
   
   </details>


## Dataset & Evaluation

- Dataset
   <details><summary>Click to expand</summary>
   
   
   | Data Type          | Description                  | Source(ZH)                                                   | Source(EN)                                                   | Source(FR)                                                   | Source(ES)                                               | Source(AR)                                                   | Source(HI)                                                   |
   | ------------------ | ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | Continue Pretrain  |                              |                                                              |                                                              |                                                              |                                                          |                                                              |                                                              |
   | Medical Books      | Medical related Books        | MedQA-books                                                  | Pile-Books                                                   | -                                                            |                                                          |                                                              |                                                              |
   | Medical Guidelines | Clinical Medicine Guide      | Chinese Medical Association                                  | [Medtron guideline](https://huggingface.co/datasets/epfl-llm/guidelines) | -                                                            |                                                          |                                                              |                                                              |
   | Medical Wiki       | Medical related wikipedia    | Wikipedia & Wikidoc                                          | Wikipedia  & Wikidoc                                         | [CLEAR - Simple Corpus for Medical French](http://natalia.grabar.free.fr/resources.php#clear) | -                                                        | -                                                            | [Hindi_health](https://www.kaggle.com/datasets/aijain/hindi-health-dataset/data?select=Symptom+Gazetteer.txt) |
   | Medical Paper      | Medical related paper        | Papers abstract                                              | PubMed Abstract                                              | [MORFITT](https://huggingface.co/datasets/qanastek/MORFITT?row=98): Pubmed-french Cochrane: [CLEAR-](http://natalia.grabar.free.fr/resources.php#clear)abs | [Mesinesp](https://zenodo.org/records/3826492)           | -                                                            | -                                                            |
   | Medical Web        | Medical related web data     | Wudao                                                        | C4                                                           | [Frenchmedmcqa](https://github.com/qanastek/FrenchMedMCQA)_train | [CoWeSe](https://zenodo.org/records/5513237)             | -                                                            | -                                                            |
   | SFT                |                              |                                                              |                                                              |                                                              |                                                          |                                                              |                                                              |
   | Medical Exam       | Medical related exams        | MedQA CExam CMB (Train Set)                                  | MedQA MedmcQA PubMedQA  (Train Set)                          | -                                                            | [Head_qa](https://huggingface.co/datasets/head_qa)_train | -                                                            | -                                                            |
   | Medical Patient    | Doctor-patient dialogue data | [HuatuoGPT-I](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) | [PMC_patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients?row=34) | -                                                            | -                                                        | [MAQA](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y2JBEZ) | -                                                            |
   | General_Replay     | General SFT Data             | Wizard & ShareGPT & Alpaca                                   | Wizard & ShareGPT & Alpaca & [Dataset List](https://huggingface.co/jondurbin/bagel-dpo-34b-v0.2#sft-data-sources) | ShareGPT & Alpaca                                            | ShareGPT & Alpaca                                        | ShareGPT & Alpaca                                            | ShareGPT & Alpaca                                            |
   | Code               | Code Data                    | [leetcode-11k](https://huggingface.co/datasets/krisfu/awesome-llm-datasets-only-Chinese) | [python_alpaca](https://huggingface.co/datasets/Vezora/Tested-22k-Python-Alpaca) | -                                                            | -                                                        | -                                                            | -                                                            |
   | Math               | Math Data                    |                                                              | [mathinstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | -                                                            | -                                                        | -                                                            | -                                                            |
   
   
   </details>

- Evaluation
   <details><summary>Click to expand</summary>
      
   [ALL test data](https://github.com/FreedomIntelligence/Medbase/tree/main/metadata/test)
   
  - EN:
    - [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) 
    - [MedMCQA](https://huggingface.co/datasets/medmcqa/viewer/default/test)
    - [PubMedQA](https://huggingface.co/datasets/pubmed_qa)
    - [MMLU-Medical](https://huggingface.co/datasets/cais/mmlu)
       - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
  - ZH:
     - [MedQA-MCMLE](https://huggingface.co/datasets/bigbio/med_qa/viewer/med_qa_zh_4options_bigbio_qa/test)
     - [CMB-single](https://huggingface.co/datasets/FreedomIntelligence/CMB)
        - Randomly sample 2,000 multiple-choice questions with single answer.
     - [CMMLU-Medical](https://huggingface.co/datasets/haonan-li/cmmlu)
        - Anatomy, Clinical_knowledge, College_medicine, Genetics, Nutrition, Traditional_chinese_medicine, Virology
     - [CExam](https://github.com/williamliujl/CMExam)
        - Randomly sample 2,000 multiple-choice questions


  - ES: [Head_qa](https://huggingface.co/datasets/head_qa)
  - FR: [Frenchmedmcqa](https://github.com/qanastek/FrenchMedMCQA)
  - HI: [MMLU_HI](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic)
    - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
  - AR: [MMLU_Ara](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)
    - Clinical knowledge, Medical genetics, Anatomy, Professional medicine, College biology, College medicine
      
  
   - Prompt: Please refer to [test generate code](https://github.com/FreedomIntelligence/Medbase/blob/main/src/process/prepare/data_process_test_qwen.py)
         
   </details>


## Results reproduction
   <details><summary>Click to expand</summary>
   
   1. Prepare Training Data
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
