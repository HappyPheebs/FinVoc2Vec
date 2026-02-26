from transformers import PretrainedConfig


class FinVoc2VecConfig(PretrainedConfig):
    """
    自定义 FinVoc2Vec 的配置类。
    这里需要特别注意 num_labels 的设置顺序：
    先设置 id2label / label2id，再设置 num_labels，避免 transformers 内部访问不存在的 id2label。
    """

    model_type = "finvoc2vec"

    def __init__(
        self,
        num_labels: int = 3,
        pooling_mode: str = "mean",
        class_weights=None,
        id2label=None,
        label2id=None,
        **kwargs,
    ):
        # 先处理并设置 id2label / label2id，避免 transformers 在设置 num_labels 时访问不到 id2label
        if id2label is None:
            id2label = {
                0: "negative",
                1: "neutral",
                2: "positive",
            }

        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        # 先设置这些映射
        self.id2label = id2label
        self.label2id = label2id

        # 其他自定义字段
        self.pooling_mode = pooling_mode
        self.class_weights = class_weights

        # 再设置 num_labels，保证此时 id2label 已存在
        # 如果从 config.json 里读取的 num_labels 和映射不一致，以映射长度为准
        self.num_labels = len(self.id2label) if num_labels is None else num_labels

        # 传递给父类，保证 transformers 内部状态一致
        super().__init__(
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            **kwargs,
        )
