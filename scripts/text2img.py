import requests
import io
import base64
from PIL import Image, PngImagePlugin
import functools
from scripts.constants import *
from random import randrange

# def img2Img(config={"name": "output"}):
#     payload = {
#         "prompt": "<lora:magic_40:1>, magic style, travel, ship, masterpiece,  high quality",
#         "negative_prompt": "EasyNegtive",
#         "sampler_name": "Euler a",
#         "cfg_scale": 7,
#         "batch_size": 1,
#         "steps": 25,
#         **config,
#     }
#     response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)

#     r = response.json()

#     for index, i in enumerate(r["images"]):
#         image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

#         png_payload = {"image": "data:image/png;base64," + i}
#         response2 = requests.post(url=f"{url}/sdapi/v1/png-info", json=png_payload)

#         pnginfo = PngImagePlugin.PngInfo()
#         info = response2.json().get("info")
#         pnginfo.add_text("parameters", info)
#         image.save("output/{}{}.png".format(config["name"], index), pnginfo=pnginfo)
#         print("{}/{}".format(index + 1, len(r["images"])))
#         print(info)


def text2Img(config={"name": "output"}):
    payload = {
        "prompt": "<lora:magic_40:1>, magic style, travel, ship, masterpiece,  high quality",
        "negative_prompt": "EasyNegtive",
        "sampler_name": "Euler a",
        "cfg_scale": 7,
        "batch_size": 1,
        "steps": 20
        # **config,
    }
    response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)

    r = response.json()

    print('res', r)

    for index, i in enumerate(r["images"]):
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

        png_payload = {"image": "data:image/png;base64," + i}
        response2 = requests.post(url=f"{url}/sdapi/v1/png-info", json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        info = response2.json().get("info")
        pnginfo.add_text("parameters", info)
        image.save("output/{}{}.png".format(config["name"], index), pnginfo=pnginfo)
        print(info)


def switchCheckpoint(checkpointName):
    option_payload = {"sd_model_checkpoint": checkpointName}

    response = requests.post(url=f"{url}/sdapi/v1/options", json=option_payload)

def getCheckpointsList():
    response = requests.get(url=f"{url}/dreambooth/checkpoints")
    installedCheckpoints = response.json()
    print('Installed Checkpoints: ')
    print(installedCheckpoints)
    return installedCheckpoints

def getIndexInList(data):
    return {
        "get": lambda index: data[index],
        "len": len(data)
    }

def getLoraComb(mainLora, subLoras):
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

def autorun():
    checkpoints = []
    # list(
    #     filter(
    #         lambda checkpoint: checkpoint not in CheckpointsBlacklist,
    #         getCheckpointsList(),
    #     )
    # )
    variables = [getLoraComb(mainLora, subLoras), getIndexInList(posPrompts), getIndexInList(samplers), getIndexInList(checkpoints)]
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
        
        text2Img(
            {
                "prompt": "{}, {}, {}".format(loraComb, posPrompt, qualityPrompt),
                "negative_prompt": negPrompt,
                "sampler": sampler,
                "batch_size": batchSize,
                "name": "{}-{}-{}-".format(checkpoint[:5], sampler, loraComb),
            }
        )

    for i in range(totalRuns):
        run1Comb(i)
        print("{}/{}".format(i + 1, totalRuns))


# autorun()
