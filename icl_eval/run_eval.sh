export CUDA_VISIBLE_DEVICES=$1
model=$2
modelname=$(basename "$model")

export TOKENIZERS_PARALLELISM=false
cd icl_eval

bash hf_open_llm.sh $model arxiv_continuation,book_continuation,c4_continuation,common_crawl_continuation,github_continuation,stackexchange_continuation,wikipedia_continuation 0 $model/result/continuation0shot-$modelname

# more
bash hf_open_llm.sh $model siqa,openbookqa,squadv2,boolq 0 $model/result/my0shot-$modelname
bash hf_open_llm.sh $model nq_open,triviaqa 5 $model/result/my5shot-$modelname

# zero-shot evaluation for Pythia evaluation tasks
bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 0 $model/result/pythia0shot-$modelname

# HF leaderboard evaluation
bash hf_open_llm.sh $model mmlu_flan_n_shot_loglikelihood_abstract_algebra,mmlu_flan_n_shot_loglikelihood_anatomy,mmlu_flan_n_shot_loglikelihood_astronomy,mmlu_flan_n_shot_loglikelihood_business_ethics,mmlu_flan_n_shot_loglikelihood_clinical_knowledge,mmlu_flan_n_shot_loglikelihood_college_biology,mmlu_flan_n_shot_loglikelihood_college_chemistry,mmlu_flan_n_shot_loglikelihood_college_computer_science,mmlu_flan_n_shot_loglikelihood_college_mathematics,mmlu_flan_n_shot_loglikelihood_college_medicine,mmlu_flan_n_shot_loglikelihood_college_physics,mmlu_flan_n_shot_loglikelihood_computer_security,mmlu_flan_n_shot_loglikelihood_conceptual_physics,mmlu_flan_n_shot_loglikelihood_econometrics,mmlu_flan_n_shot_loglikelihood_electrical_engineering,mmlu_flan_n_shot_loglikelihood_elementary_mathematics,mmlu_flan_n_shot_loglikelihood_formal_logic,mmlu_flan_n_shot_loglikelihood_global_facts,mmlu_flan_n_shot_loglikelihood_high_school_biology,mmlu_flan_n_shot_loglikelihood_high_school_chemistry,mmlu_flan_n_shot_loglikelihood_high_school_computer_science,mmlu_flan_n_shot_loglikelihood_high_school_european_history,mmlu_flan_n_shot_loglikelihood_high_school_geography,mmlu_flan_n_shot_loglikelihood_high_school_government_and_politics,mmlu_flan_n_shot_loglikelihood_high_school_macroeconomics,mmlu_flan_n_shot_loglikelihood_high_school_mathematics,mmlu_flan_n_shot_loglikelihood_high_school_microeconomics,mmlu_flan_n_shot_loglikelihood_high_school_physics,mmlu_flan_n_shot_loglikelihood_high_school_psychology,mmlu_flan_n_shot_loglikelihood_high_school_statistics,mmlu_flan_n_shot_loglikelihood_high_school_us_history,mmlu_flan_n_shot_loglikelihood_high_school_world_history,mmlu_flan_n_shot_loglikelihood_human_aging,mmlu_flan_n_shot_loglikelihood_human_sexuality,mmlu_flan_n_shot_loglikelihood_international_law,mmlu_flan_n_shot_loglikelihood_jurisprudence,mmlu_flan_n_shot_loglikelihood_logical_fallacies,mmlu_flan_n_shot_loglikelihood_machine_learning,mmlu_flan_n_shot_loglikelihood_management,mmlu_flan_n_shot_loglikelihood_marketing,mmlu_flan_n_shot_loglikelihood_medical_genetics,mmlu_flan_n_shot_loglikelihood_miscellaneous,mmlu_flan_n_shot_loglikelihood_moral_disputes,mmlu_flan_n_shot_loglikelihood_moral_scenarios,mmlu_flan_n_shot_loglikelihood_nutrition,mmlu_flan_n_shot_loglikelihood_philosophy,mmlu_flan_n_shot_loglikelihood_prehistory,mmlu_flan_n_shot_loglikelihood_professional_accounting,mmlu_flan_n_shot_loglikelihood_professional_law,mmlu_flan_n_shot_loglikelihood_professional_medicine,mmlu_flan_n_shot_loglikelihood_professional_psychology,mmlu_flan_n_shot_loglikelihood_public_relations,mmlu_flan_n_shot_loglikelihood_security_studies,mmlu_flan_n_shot_loglikelihood_sociology,mmlu_flan_n_shot_loglikelihood_us_foreign_policy,mmlu_flan_n_shot_loglikelihood_virology,mmlu_flan_n_shot_loglikelihood_world_religions 5 $model/result/mmlu5shot-$modelname
bash hf_open_llm.sh $model hellaswag 10 $model/result/hellaswag10shot-$modelname
bash hf_open_llm.sh $model arc_challenge 25 $model/result/arcc25shot-$modelname
bash hf_open_llm.sh $model truthfulqa 0 $model/result/truthfulqa0shot-$modelname

# others
bash hf_open_llm.sh $model nq_open 32 $model/result/nq_open32shot-$modelname
bash hf_open_llm.sh $model boolq 32 $model/result/boolq32shot-$modelname
bash hf_open_llm.sh $model gsm8k 8 $model/result/gsm8k8shot-$modelname
