import telebot
import torch
from torch.autograd import Variable

from config.config import TOKEN
from model.network import Net
from model.utils import tensor_save_bgrimage, preprocess_batch, tensor_load_rgbimage


print(TOKEN)
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def welcome_start(message):
    bot.send_message(message.chat.id, 'Hello')

@bot.message_handler(commands=["content"])
def handle_send_photo(message):
    print('!!!')
    file = open('tmp/output.jpg', 'rb')
    bot.send_photo(message.chat.id, file)
    file.close()


def prepare_model():
    style_model = Net(ngf=128)
    model_dict = torch.load('21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    return style_model


def prepare_inputs():
    content_image = tensor_load_rgbimage('city.jpg', size=512,
                                         keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage('wall.jpg', size=512).unsqueeze(0)
    style = preprocess_batch(style)
    return content_image, style


def get_output_image(content_image, style):
    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    tensor_save_bgrimage(output.data[0], 'tmp/output.jpg', False)


style_model = prepare_model()
content_image, style_image = prepare_inputs()
get_output_image(content_image, style_image)
print('Done')
bot.polling()
