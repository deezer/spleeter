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

title = "Spleeter"
description = "demo for Spleeter by Deezer. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://research.deezer.com/projects/spleeter.html'>Spleeter: a Fast and Efficient Music Source Separation Tool with Pre-Trained Models</a> | <a href='https://github.com/deezer/spleeter'>Github Repo</a></p>"
examples = [
    ["audio_example.mp3"]
]

gr.Interface(spleeter, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()