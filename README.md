# ChotuGPT
This repository has the implementation of the Transformer architecture from scratch. It consists of an Encoder and Decoder, including self attention and cross attention making it capable of generating guidance influenced text

<img src="https://github.com/PraNavKumAr01/ChotuGPT/blob/main/transformer.jpeg" width="500" height="500">

# Architecture Details
## Encoder
The input tokens are first encoded, then it is turned into an en embedding and then concatened with the positional embedding <br />
Then these embeddings pass into the Encoder block which consists of the following - 

---- A multi attention head which calculates the attention scores by masking the future tokens of the current token.<br />
---- A Adding and normalizing layer, We use batch norm in this for normalization<br />
---- A Feed forward layer which lets the token communicate after the attention scores have been calculated<br />
---- The outputs of the encoder are then sent to the second attention head in the decoder to calculate cross attention scores

## Decoder
The outputs shifted right or basically the training data is first encoded, then it is turned into an en embedding and then concatened with the positional embedding <br />
Then these embeddings pass into the Decoder block which consists of the following :

---- A masked multi attention head which calculates the attention scores by masking the future tokens of the current token.<br />
---- A Adding and normalizing layer, We use batch norm in this for normalization<br />
---- A Feed forward layer which lets the token communicate after the attention scores have been calculated<br />

---- A multi attention head which receives the output embeddings from the encoder and calculates the cross attention scores<br />
---- A Adding and normalizing layer, We use batch norm in this for normalization<br />
---- A Feed forward layer which lets the token communicate after the attention scores have been calculated<br />
Now these same are repeated in multiple number of blocks to get the final output which is passed through another linear layer and a softmax to get the probabilites

# Model Details

I have trained a decoder only model with 43 Million parameters for around 30 thousand epochs. Got a training loss of around 0.6 and validation loss of around 0.9

This model is able to form basic english words and sentences but still lacks the ability to perfectly form sentences that make sense

I have also included the complete code and architecture of the encoder + decoder transformer, but as it reached almost 100M parameters, its almost impossible to train on free GPU's<br />

But if any of you do have the resources !!GO AHEAD!!

# Repository Details

The `gpttrain.ipynb` contains the data loading and training code. You can run the same code and tune the hyperparameters by yourself. 

The `completegpt.ipynb` contains the entire code of creating and loading data, as well as the complete architecture and training loop for the encoder + decoder transformer

The `tokenizer.py` file has the tokenizer based on the Byte Pair Algorithm, but has not yet been impemented along with the transformer model

The `config.py` file has the definitions of the GPT Model and Hyperparameters

The `chars_list.pk1` file has the vocabulary of the text that this model has been trained on, This is needed during inference

The `generate.py` file has the code for running inference, testing and sampling from the model 

# Running the model

To run the model follow these steps

First clone the github repository

`!git clone https://github.com/PraNavKumAr01/ChotuGPT.git`<br />

 Get the weights from huggingface
 
`!git clone https://huggingface.co/PK03/GPT43M_30K`

Then cd into the github repository to get the `generate.py` file

`%cd /content/ChotuGPT`

Run the `generate.py` file

Give the huggingface weights path to `--model_path` and how long you want the response to be generated to `--max_new_tokens`

`!python generate.py --model_path /content/GPT43M_30K/gpt43M_30k.pth --max_new_tokens 600`

Or you can directly run this colab notebook

[Colab Notebook](https://colab.research.google.com/drive/1bN6eHQW9TgDBsx8gE8MO9icEkNsDRbiV?usp=sharing)
