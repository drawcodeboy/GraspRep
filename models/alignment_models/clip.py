from transformers import CLIPConfig, CLIPModel
from transformers.models.clip.modeling_clip import CLIPPretrainedModel

class PointCLIPModel(CLIPPretrainedModel):
    def __init__(self,
                 config: CLIPConfig):
        super().__init__()

    