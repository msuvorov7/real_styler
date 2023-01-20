import json
import logging
import os
import numpy as np
import onnxruntime
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from PIL import Image
from io import BytesIO


log = logging.getLogger(__name__)


async def welcome_start(message):
    """Aiogram helper handler."""

    await message.answer('Hello!\nSend Photo for beginning...')


async def content_handler(message: types.message):
    """Aiogram handler only for text messages."""
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_mosaic.onnx')

    content_image = np.asarray(Image.open(image).convert('RGB'), dtype=np.float32)
    content_image = content_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    model_input = {model.get_inputs()[0].name: content_image}
    model_output = model.run(None, model_input)

    result = Image.fromarray(model_output[0][0].clip(0, 255).transpose(1, 2, 0).astype('uint8'))
    result_bytes = BytesIO()
    result.save(result_bytes, 'jpeg')
    result_bytes.seek(0)

    await message.answer_photo(result_bytes)


# Functions for Yandex.Cloud
async def register_handlers(dp: Dispatcher):
    """Registration all handlers before processing update."""

    dp.register_message_handler(welcome_start, commands=['start'])
    dp.register_message_handler(content_handler, content_types=['photo'])

    log.debug('Handlers are registered.')


async def process_event(event, dp: Dispatcher):
    """
    Converting an Yandex.Cloud functions event to an update and
    handling tha update.
    """

    update = json.loads(event['body'])
    log.debug('Update: ' + str(update))

    Bot.set_current(dp.bot)
    update = types.Update.to_object(update)
    await dp.process_update(update)


async def handler(event, context):
    """Yandex.Cloud functions handler."""

    if event['httpMethod'] == 'POST':

        # Bot and dispatcher initialization
        bot = Bot(os.environ.get('TELEGRAM_BOT_TOKEN'))
        dp = Dispatcher(bot)

        await register_handlers(dp)
        await process_event(event, dp)

        return {'statusCode': 200, 'body': 'ok'}
    return {'statusCode': 405}
