import json
import logging
import os
import numpy as np
import onnxruntime
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from PIL import Image
from io import BytesIO

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


log = logging.getLogger(__name__)

keyboards = [
    InlineKeyboardButton(text='mosaic', callback_data='mosaic_style'),
    InlineKeyboardButton(text='udnie', callback_data='udnie_style'),
    InlineKeyboardButton(text='lines', callback_data='lines_style'),
    InlineKeyboardButton(text='candy', callback_data='candy_style'),
    InlineKeyboardButton(text='scream', callback_data='scream_style'),
    InlineKeyboardButton(text='night', callback_data='night_style'),
]

style_menu = InlineKeyboardMarkup(row_width=3)
style_menu.add(*keyboards)


class UploadPhotoForStyle:
    def __init__(self):
        self.state = None

    def set_state(self, name: str):
        self.state = name

    def get_state(self):
        return self.state


style_state = UploadPhotoForStyle()


async def welcome_start(message):
    await message.answer('Hello!\nChoose style:', reply_markup=style_menu)


async def process_mosaic_style(callback_query: types.CallbackQuery):
    style_state.set_state('mosaic')
    await callback_query.message.answer('Activated mosaic style\nSend photo')


async def process_udnie_style(callback_query: types.CallbackQuery):
    style_state.set_state('udnie')
    await callback_query.message.answer('Activated udnie style\nSend photo')


async def process_lines_style(callback_query: types.CallbackQuery):
    style_state.set_state('lines')
    await callback_query.message.answer('Activated lines style\nSend photo')


async def process_candy_style(callback_query: types.CallbackQuery):
    style_state.set_state('candy')
    await callback_query.message.answer('Activated candy style\nSend photo')


async def process_scream_style(callback_query: types.CallbackQuery):
    style_state.set_state('scream')
    await callback_query.message.answer('Activated scream style\nSend photo')


async def process_night_style(callback_query: types.CallbackQuery):
    style_state.set_state('night')
    await callback_query.message.answer('Activated night style\nSend photo')


async def process_mosaic_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_mosaic.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


async def process_udnie_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_udnie.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


async def process_lines_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_lines.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


async def process_candy_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_candy.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


async def process_scream_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_scream.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


async def process_night_photo(message: types.Message):
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    model = onnxruntime.InferenceSession('models/model_night.onnx')
    result_bytes = get_styled_photo(image, model)

    await message.answer_photo(photo=result_bytes)


def get_styled_photo(image, model):
    content_image = np.asarray(Image.open(image).convert('RGB'), dtype=np.float32)
    content_image = content_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

    model_input = {model.get_inputs()[0].name: content_image}
    model_output = model.run(None, model_input)

    result = Image.fromarray(model_output[0][0].clip(0, 255).transpose(1, 2, 0).astype('uint8'))
    result_bytes = BytesIO()
    result.save(result_bytes, 'jpeg')
    result_bytes.seek(0)

    return result_bytes


# Functions for Yandex.Cloud
async def register_handlers(dp: Dispatcher):
    """Registration all handlers before processing update."""

    dp.register_callback_query_handler(process_mosaic_style, lambda item: item.data == 'mosaic_style')
    dp.register_callback_query_handler(process_udnie_style, lambda item: item.data == 'udnie_style')
    dp.register_callback_query_handler(process_lines_style, lambda item: item.data == 'lines_style')
    dp.register_callback_query_handler(process_candy_style, lambda item: item.data == 'candy_style')
    dp.register_callback_query_handler(process_scream_style, lambda item: item.data == 'scream_style')
    dp.register_callback_query_handler(process_night_style, lambda item: item.data == 'night_style')

    log.debug('Callback queries are registered.')

    dp.register_message_handler(welcome_start, commands=['start'])
    dp.register_message_handler(process_mosaic_photo, lambda item: style_state.get_state() == 'mosaic', content_types=['photo'])
    dp.register_message_handler(process_udnie_photo, lambda item: style_state.get_state() == 'udnie', content_types=['photo'])
    dp.register_message_handler(process_lines_photo, lambda item: style_state.get_state() == 'lines', content_types=['photo'])
    dp.register_message_handler(process_candy_photo, lambda item: style_state.get_state() == 'candy', content_types=['photo'])
    dp.register_message_handler(process_scream_photo, lambda item: style_state.get_state() == 'scream', content_types=['photo'])
    dp.register_message_handler(process_night_photo, lambda item: style_state.get_state() == 'night', content_types=['photo'])

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
