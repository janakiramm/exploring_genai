import bentoml
from bentoml.io import Image, JSON

from sdargs import SDArgs

bento_model = bentoml.diffusers.get("sd2.1:latest")
sd21_runner = bento_model.to_runner(name="sd21-runner")

svc = bentoml.Service("stable-diffusion-21", runners=[sd21_runner])


@svc.api(input=JSON(pydantic_model=SDArgs), output=Image())
async def txt2img(input_data):
    kwargs = input_data.dict()
    res = await sd21_runner.async_run(**kwargs)
    images = res[0]
    return images[0]