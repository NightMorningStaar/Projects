# from flask import Flask, request
# import argparse
# from transformers import AutoModelWithLMHead, AutoTokenizer
# import torch
# # from twilio.twiml.messaging_response import MessagingResponse
#
#
# parser = argparse.ArgumentParser(
#     description="Process chatbot variables. for help run python bot.py -h"
# )
# parser.add_argument(
#     "-m", "--model", type=str, default="medium", help="Size of DialoGPT model"
# )
# parser.add_argument(
#     "-s",
#     "--steps",
#     type=int,
#     default=7,
#     help="Number of steps to run the Dialogue System for",
# )
#
# args = parser.parse_args()
# tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{args.model}")
# model = AutoModelWithLMHead.from_pretrained(f"microsoft/DialoGPT-{args.model}")
#
# app = Flask(__name__)
#
#
# @app.route("/bot", methods=["POST"])
# def bot():
#     for step in range(args.steps):
#         incoming_msg = request.values.get("Body", "").lower()
#
#         new_user_input_ids = tokenizer.encode(
#             incoming_msg + tokenizer.eos_token, return_tensors="pt"
#         )
#         bot_input_ids = (
#             torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
#             if step > 0
#             else new_user_input_ids
#         )
#         chat_history_ids = model.generate(
#             bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
#         )
#
#         resp = MessagingResponse()
#         msg = resp.message()
#         msg.body(
#             f"{tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)}"
#         )
#         return str(resp)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
# """
#
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# """



from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from flask import Flask, request, jsonify, json
import argparse
# from twilio.twiml.messaging_response import MessagingResponse

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

parser = argparse.ArgumentParser(
    description="Process chatbot variables. for help run python bot.py -h"
)
parser.add_argument(
    "-m", "--model", type=str, default="medium", help="Size of DialoGPT model"
)
parser.add_argument(
    "-s",
    "--steps",
    type=int,
    default=7,
    help="Number of steps to run the Dialogue System for",
)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{args.model}")
model = AutoModelWithLMHead.from_pretrained(f"microsoft/DialoGPT-{args.model}")


# Let's chat for 5 lines
app = Flask(__name__)

@app.route("/")
def bot():
    # if request.method == "POST":

        for step in range(5):
            input = request.args.get('message')
            # incoming_msg = request.values.get("Body", "").lower()
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last ouput tokens from bot
            r = "DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

            return jsonify(r)

        # resp = MessagingResponse()
        # msg = resp.message()
        # msg.body(f"{tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)}")

        # return str(resp)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False, threaded = True)

# @app.route("/", methods=["POST", "GET"])
# def bot(input):
#
#     for step in range(5):
#         # incoming_msg = request.values.get("Body", "").lower()
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
#
#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
#
#         # generated a response while limiting the total chat history to 1000 tokens,
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#
#         # pretty print last ouput tokens from bot
#         return "DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
#
#         # resp = MessagingResponse()
#         # msg = resp.message()
#         # msg.body(f"{tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)}")
#
#         # return str(resp)