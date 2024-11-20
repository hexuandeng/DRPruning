export CUDA_VISIBLE_DEVICES=$1
model=$2
modelname=$(basename "$(dirname "$model")")-$(basename "$model")

export TOKENIZERS_PARALLELISM=false
cd icl_eval

bash hf_open_llm.sh $model xnli_ar,xnli_en,xnli_ru,xnli_th,xnli_tr,xnli_zh 0 $model/result/xnli0shot-$modelname
bash hf_open_llm.sh $model xstorycloze_ar,xstorycloze_en,xstorycloze_ru,xstorycloze_zh 0 $model/result/xstorycloze0shot-$modelname
bash hf_open_llm.sh $model xwinograd_en,xwinograd_jp,xwinograd_ru,xwinograd_zh 0 $model/result/xwinograd0shot-$modelname
bash hf_open_llm.sh $model paws_en,paws_ja,paws_ko,paws_zh 0 $model/result/paws0shot-$modelname

bash hf_open_llm.sh $model xnli_ar,xnli_en,xnli_ru,xnli_th,xnli_tr,xnli_zh 4 $model/result/xnli4shot-$modelname
bash hf_open_llm.sh $model xstorycloze_ar,xstorycloze_en,xstorycloze_ru,xstorycloze_zh 4 $model/result/xstorycloze4shot-$modelname
bash hf_open_llm.sh $model xwinograd_en,xwinograd_jp,xwinograd_ru,xwinograd_zh 4 $model/result/xwinograd4shot-$modelname
bash hf_open_llm.sh $model paws_en,paws_ja,paws_ko,paws_zh 4 $model/result/paws4shot-$modelname
