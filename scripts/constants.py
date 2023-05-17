# Stable-diffusion-webui hosting address WebUI的地址
# url = "http://10.37.150.64:7860"
url = "http://127.0.0.1:7860"

# Checkpoints to IGNORE 需要忽略的Checkpoint（黑名单以外的Checkpoint都会跑）
CheckpointsBlacklist = [
  "3dchanStyle_v1.ckpt [9698c46f61]",
  "adi3d_dreambooth_100.ckpt [4b59ac4a71]",
  "adi3d_dreambooth_1000.ckpt [75c1abb953]",
  "adi3d_dreambooth_3000.ckpt [dc8b8ff86d]",
  "adiHM_dreambooth_3d.ckpt [8c44e12e18]",
  "anything-v4.5-pruned-fp16.ckpt [f773383dbc]",
  "chilloutmix_NiPrunedFp32Fix.safetensors [fc2511737a]",
  "cuteRichstyle15_cuteRichstyle.ckpt [24bc802fc5]",
  "ddicon_v20.ckpt [1243b42854]",
  "deliberate_v2.safetensors [9aba26abdf]",
  "dreamRealComic_v15.safetensors [a829be4131]",
  "emojiModel_v10.ckpt [b717b74f16]",
  "hollieMengert_v1.ckpt [2c4c9a75f6]",
  "isometricFuture_isometricFutures10.safetensors [aa9c45d00a]",
  "knollingcase_v1.ckpt [cf836e65a7]",
  "magic_dreambooth_10000.ckpt [7354d126ed]",
  "MAGIFACTORYTShirt_magifactoryTShirtModel.safetensors [ea0087f1e6]",
  "marioBrosLuxxx_v10.ckpt [659e883905]",
  # "revAnimated_v122.safetensors [f8bb2922e1]"
]

# Prompts for quality 质量Prompt
qualityPrompt = "simple background, high quality, best quality, masterpiece"
# Postive prompts 正向Prompt
posPrompts = [
    "shopping, mall, store, clothes, shoes, hat",
    "shopping, mall, store, clothes, shoes, hat, 3D, simple design, sandbox, clean",
    "shopping, mall, store, clothes, shoes, hat, (Isometric perspective:1.5),(Pixar style:1.2), clean, 3D,Disney,Panoramic photo,White background, solid background,Global illumination, raytracing, modeling,HDR,octane render,unreal render,behance,dribbble,artstation,best quality, 8k"
    ]
# Negtive prompts 反向Prompt
negPrompt = "EasyNegative,low resolution,bad composition,blurry image,disfigured,oversaturated, blind, lowres, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ((black and white))"

# Lora combinations Lora组合
# Main Lora (and activation words) 主要用到的Lora（及触发词），大括号务必保留
mainLora = "<lora:magic_40:{:.2f}>, magic style"
# Sub Lora (and activation words) 辅助使用的Lora（及触发词）），大括号务必保留
subLoras = ["<lora:mScene_v10:{:.2f}>"]
# num of lora combinations: loraCombCount^(num of subLoras)
loraCombCount = 2

# Samplers 采样器
samplers = ["Euler a", "DMP adaptive", "DPM++ SDE Karras"]

# Batch size 每批生成数量
batchSize = 2
