# download ApolloCorpus

cd metadata
wget https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus/resolve/main/ApolloCorpus.zip
unzip ApolloCorpus.zip

# Prepare Data for Mix training
mkdir mixTrain


cd train/pretrain
# Mixtraining Only use QA pairs in Pretrain
for file in *; do
    if [[ $file == *_qa.json ]]; then
        cp "$file" "../mixTrain/"
    fi
done
cd ../

# copy all file from sft to mix_train
mv sft/* mixTrain/

# merge all the file from mix_train directory to json
python merge_json_train.py
cd ../


