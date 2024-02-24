# Copied from https://github.com/MhmdSaiid/RuleBert/blob/main/download_datasets.sh
# Visit https://github.com/MhmdSaiid the generation code of this dataset
mkdir -p data
wget -O data/rulebert_data.zip https://zenodo.org/record/5644677/files/RuleBERT_Datasets.zip?download=1 
unzip data/rulebert_data.zip -d data
rm data/rulebert_data.zip