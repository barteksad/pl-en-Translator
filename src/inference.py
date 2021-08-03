from config import CFG

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

import gradio as gr

import os

def predict(text):
  text = tokenizer(text, return_tensors='pt')
  output = model.generate(**text)
  output = tokenizer.decode(output[0], skip_special_tokens=True)

  return output


if __name__ == '__main__':
    global model
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if os.path.exists('src/model_checkpoints/finetuned_model_best_loss'):
      model = AutoModelForSeq2SeqLM.from_pretrained('src/model_checkpoints/finetuned_model_best_loss')
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)
    model.eval()

    iface = gr.Interface(fn=predict, inputs="text", outputs="text", examples=[['Jak masz na imię?']])
    iface.launch(share=True)
