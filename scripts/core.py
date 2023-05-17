from modules.txt2img import txt2img
from modules import sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html


default_txt2img_config = {
    'id_task': 'task()',
    'prompt': '',
    'negative_prompt': '',
    'prompt_styles': [],
    'steps': 20,
    'sampler_index': 0,
    'restore_faces': False,
    'tiling': False,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'seed': -1,
    'subseed': -1,
    'subseed_strength': 0,
    'seed_resize_from_h': 0,
    'seed_resize_from_w': 0,
    'seed_enable_extras': False,
    'height': 512,
    'width': 512,
    'enable_hr': False,
    'denoising_strength': 0.7,
    'hr_scale': 2,
    'hr_upscaler': 'Latent',
    'hr_second_pass_steps': 0,
    'hr_resize_x': 0,
    'hr_resize_y': 0,
    'override_settings_texts': [] 
}
args = []


def doTxt2Img(config):
    config = { **default_txt2img_config, **config }
    print('config', config)

    final_args = [*list(config.values()), *args]
    imgs, info, _, _ = txt2img(*final_args)
    print(imgs, info)
    return imgs, info


def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, override_settings_texts, *args):
    override_settings = create_override_settings_dict(override_settings_texts)
    print('args', args)

    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        override_settings=override_settings,
    )

    # p.scripts = modules.scripts.scripts_txt2img
    # p.script_args = args

    processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)
