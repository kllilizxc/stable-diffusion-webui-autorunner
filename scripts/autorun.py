import modules.scripts as scripts
import gradio as gr
import os
from PIL import Image, PngImagePlugin
import functools
from modules import script_callbacks, sd_models, sd_samplers
from modules.shared import opts
from random import randrange
# from scripts.text2img import *
from scripts.core import doTxt2Img

sampler_names = [sampler.name for sampler in sd_samplers.samplers]
cList = []
interrupt = False
running = False
lastGeneratedImg = None

def interruptRun():
    global interrupt
    interrupt = True

def txt2list(txt):
    return txt.rstrip('\n').split(';')

def switchCheckpoint(checkpointName):
    opts.sd_model_checkpoint = checkpointName
    sd_models.reload_model_weights()

def getIndexInList(data):
    return {
        "get": lambda index: data[index],
        "len": len(data)
    }

def getLoraComb(mainLora, subLoras, loraCombCount):
    def getLoraPrompt(index):
        mainRatio = 0.5 + randrange(50) / 100
        ratioSum = mainRatio
        def getter(a, b):
            nonlocal ratioSum
            if b == subLoras[-1]:
                subRatio = 1 - ratioSum
            else:
                subRatio = randrange((1 - ratioSum) * 100) / 100
                ratioSum = ratioSum + subRatio
            return "{}, {}".format(a, b.format(subRatio))
        return functools.reduce(getter, [mainLora.format(mainRatio), *subLoras])
    return {
            "get": getLoraPrompt,
            "len": pow(loraCombCount, len(subLoras))
        }

def generate(checkpoints, output, qualityPrompt, posPrompts, negPrompt, mainLora, subLoras, loraCombCount, samplers, batchSize):
    global interrupt
    global running
    global lastGeneratedImg
    running = True
    checkpoints = txt2list(checkpoints)
    posPrompts = txt2list(posPrompts)
    samplers = txt2list(samplers)
    subLoras = txt2list(subLoras)
    loraCombCount = int(loraCombCount)
    batchSize = int(batchSize)

    if not os.path.exists(output):
        os.mkdir(output)
    
    print(subLoras)
    variables = [getLoraComb(mainLora, subLoras, loraCombCount), getIndexInList(posPrompts), getIndexInList(samplers), getIndexInList(checkpoints)]
    currentCheckpoint = ""
    totalRuns = functools.reduce(lambda a, b: a * b["len"], [1, *variables])
    print("Total Runs: ", totalRuns)

    def run1Comb(index):
        runIndex = index
        indexList = []

        for variable in variables:
            varIndex = runIndex % variable["len"]
            runIndex = int(runIndex / variable["len"])
            indexList.append(varIndex)

        [loraComb, posPrompt, sampler, checkpoint] = [
            variable["get"](indexList[index]) for index, variable in enumerate(variables)
        ]

        if checkpoint != currentCheckpoint:
            switchCheckpoint(checkpointName=checkpoint)

        images, info = doTxt2Img({
            "prompt": "{}, {}, {}".format(loraComb, posPrompt, qualityPrompt),
            "negative_prompt": negPrompt,
            "sampler_index": sampler_names.index(sampler),
            "batch_size": batchSize,
        })
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", info)
        for i, image in enumerate(images):
            image.save("{}/{}{}.png".format(output, "{}-{}-{}-".format(checkpoint[:5], sampler, loraComb), i), pnginfo=pnginfo)
            lastGeneratedImg = image
        print(info)

    for i in range(totalRuns):
        if interrupt: 
            interrupt = False
            print('Interrupted!')
            break
        run1Comb(i)
        print("{}/{}".format(i + 1, totalRuns))
    
    running = False

# def multiInput(label, lines=1):
#     inputs = []
#     with gr.Blocks() as multi_input:
#         with gr.Row():
#             input = gr.Textbox(value="", label=label, lines=lines)
#             btn = gr.Button(value="Add")
#             btn.click(lambda _: [*inputs, input], inputs=[], outputs=[inputs])
#         with gr.Blocks():
#             dataset = gr.Dataset(components=[gr.Textbox(visible=False)], samples=inputs)
#             dataset.click(lambda arg: print(arg))
#         return inputs

def on_ui_tabs():
    global lastGeneratedImg
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column():
                checkpoints = gr.Textbox(value=";\n".join(list(map(lambda m: m.title, sd_models.checkpoints_list.values()))), label="Checkpoints", lines=10)
                output = gr.Textbox(value=os.path.abspath("../sd-output"), label="Output")
                qualityPrompt = gr.Textbox(value=" high quality, best quality, masterpiece", label="Quality Prompt", lines=1)
                posPrompts = gr.Textbox(value="1girl; 2girls", label="Postive Prompts", lines=1)
                negPrompt = gr.Textbox(value="EasyNegative", label="Negtive Prompt", lines=1)
                mainLora = gr.Textbox(value="<lora:magic_40:{:.2f}>, magic style", label="Main Lora", lines=1)
                subLoras = gr.Textbox(value="<lora:mScene_v10:{:.2f}>", label="Sub Loras", lines=1)
                loraCombCount = gr.Textbox(value="2", label="Lora comb count", lines=1)
                samplers = gr.Textbox(value=";".join(sampler_names), label="Samplers", lines=1)
                batchSize = gr.Textbox(value="1", label="Batch size", lines=1)
                with gr.Row():
                    generateBtn = gr.Button(value="Generate")
                    config = [ checkpoints, output, qualityPrompt, posPrompts, negPrompt, mainLora, subLoras, loraCombCount, samplers, batchSize ]
                    generateBtn.click(generate, inputs=[*config], outputs=[])
                    interruptBtn = gr.Button(value="Interrupt")
                    interruptBtn.click(interruptRun, inputs=[], outputs=[])
                # TODO: add more UI components (cf. https://gradio.app/docs/#components)
            with gr.Blocks():
                im = gr.Image(value=lastGeneratedImg, type='pil')
            return [(ui_component, "Auto Runner", "sd_auto_runner")]

script_callbacks.on_ui_tabs(on_ui_tabs)