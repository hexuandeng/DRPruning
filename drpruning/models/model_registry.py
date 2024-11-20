from drpruning.models.composer_llama import ComposerMosaicLlama
from drpruning.models.composer_llama3 import ComposerMosaicLlama3
from drpruning.models.composer_pythia import ComposerMosaicPythia
from drpruning.models.composer_qwen2 import ComposerMosaicQwen2

COMPOSER_MODEL_REGISTRY = {
    'mosaic_llama': ComposerMosaicLlama,
    'mosaic_llama2': ComposerMosaicLlama,
    'mosaic_pythia': ComposerMosaicPythia,
    'mosaic_together': ComposerMosaicPythia,
    'mosaic_llama3': ComposerMosaicLlama3,
    'mosaic_qwen2': ComposerMosaicQwen2,
}
