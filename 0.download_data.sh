# download ApolloCorpus
mkdir metadata
cd metadata
wget https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus/resolve/main/ApolloCorpus.zip
unzip ApolloCorpus.zip
cd train/pretrain

qa_dir="qa"
pretrain_sft_dir="pretrain_sft"

if [ ! -d "$qa_dir" ]; then
    mkdir -p "$qa_dir"
fi

if [ ! -d "$pretrain_sft_dir" ]; then
    mkdir -p "$pretrain_sft_dir"
fi

for file in *; do
    if [[ $file == *_qa.json ]]; then
        mv "$file" "$qa_dir/"
    elif [[ $file == *_text.json ]]; then
        mv "$file" "$pretrain_sft_dir/"
    fi
done
mv pretrain_sft/ ../
mv qa/ ../
cd ../
rm pretrain

mv sft/ all_sft/

