from spleeter.separator import Separator
import gradio as gr
import shutil

def spleeter(aud, instrument):
  separator = Separator('spleeter:2stems')
  try:
    shutil.rmtree("output")
  except FileNotFoundError:
    pass
  separator.separate_to_file(aud.name, "output/", filename_format="audio_example/{instrument}.wav")
  return f"./output/audio_example/{instrument}.wav"

inputs = [
          gr.inputs.Audio(label="Input Audio", type="file"),
          gr.inputs.Radio(label="Instrument", choices=["vocals", "accompaniment"], type="value")
]
outputs =  gr.outputs.Audio(label="Output Audio", type="file")

title = "wav2vec 2.0"
description = "demo for Facebook AI wav2vec 2.0 using Hugging Face transformers. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2006.11477'>wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations</a> | <a href='https://github.com/pytorch/fairseq'>Github Repo</a> | <a href='https://huggingface.co/facebook/wav2vec2-base-960h'>Hugging Face model</a></p>"
examples = [
    ["audio_example.mp3"]
]

gr.Interface(spleeter, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)