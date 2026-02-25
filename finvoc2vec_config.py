from transformers import PretrainedConfig

class FinVoc2VecConfig(PretrainedConfig):
    
    model_type = "finvoc2vec"

    def __init__(self, num_labels=3, pooling_mode="mean", class_weights=None, **kwargs):
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode
        self.class_weights = class_weights
        super().__init__(**kwargs)