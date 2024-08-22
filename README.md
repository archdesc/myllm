# myllm
从零开始构建语言模型。Create a large language model from scratch. 
基于 https://github.com/jiahe7ay/MINI_LLM 修改的。
目的是学习，探索大模型的构建，语言模型的不同方向，并打造一个10亿参数的基础模型。

## 目录

- [简介](#简介)
- [特性](#特性)
- [安装](#安装)
- [使用](#使用)
- [Project Structure](#project-structure)
- [Transformer Techniques](#transformer-techniques)
- [参考资料及项目](#参考资料及项目)
- [Contributing](#contributing)
- [License](#license)
  
## 简介

myLLM is designed for educational purposes to understand and implement the core components of language models based on transformer architectures. The project covers:

- Tokenization
- Embeddings
- Encoder-Decoder models
- Attention mechanisms
- Training and fine-tuning techniques

## 特性
- 可以自己选择自己要的huggingface模型，调节各种参数.
- 目前代码里面用的zephyr-7b-beta模型，但是修改了参数，改为了1B模型。
- 稍微修改一下可以使用modescope模型。
- **Custom Tokenizer**: Build and train your own tokenizer.
- **Embedding Layer**: Learn how to create and use embedding layers.
- **Transformer Architecture**: Implement encoder and decoder components.
- **Attention Mechanisms**: Explore different attention mechanisms, including self-attention and cross-attention.
- **Training Pipeline**: Set up a training pipeline for your model.
- **Fine-Tuning**: Techniques to fine-tune the model on specific tasks.

## Installation

## 安装

### Conda

```conda create -n myllm python=3.10
conda activate myllm
git clone https://github.com/archdesc/myllm
cd myllm
pip install -r requirement.txt
python z-pre.py
```

### Poetry

## 使用

To train and test your LLM, follow these steps:

1. **Tokenization**: Tokenize your dataset using the custom tokenizer.

   ```python
   from tokenizer import CustomTokenizer
   tokenizer = CustomTokenizer()
   tokenizer.train('path_to_dataset')
   ```

2. **Model Training**: Train the transformer model.

   ```python
   from models.GPT import GPTLanguageModel
   model = GPTLanguageModel(vocab_size=vocab_size, device=device)
   model._train(epochs=200, learning_rate=3e-4, eval_iters=100)
   ```

3. **Inference**: Use the trained model for inference.
   ```python
   from utils.helpers import load_model
   model = load_model('path_to_trained_model')
   result = model.generate('your_encoded_input_text')
   print(result)
   ```

## Project Structure

The project structure is organized as follows:

```
myLLM/
├── data/
├── models/
│   └── ...
├── tokenization/
│   ├── tokenizer.py
│   └── ...
├── utils/
│   ├── helpers.py
│   └── ...
├── README.md
└── requirements.txt
```

## Transformer Techniques

This project explores various techniques in transformer models, including:

- **Positional Encoding**: Adding positional information to the embeddings.
- **Multi-Head Attention**: Implementing and understanding multi-head attention mechanisms.
- **Layer Normalization**: Using layer normalization to stabilize training.
- **Feed-Forward Networks**: Incorporating feed-forward neural networks within the transformer blocks.
- **Residual Connections**: Implementing residual connections to improve gradient flow.

## 参考资料及项目
- 数据
  - 来源
  - 预处理
- tokenizer
  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
Feel free to customize this README to better fit your project's specifics and your preferences!
