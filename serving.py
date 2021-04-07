import json
import requests
import tensorflow as tf
import tensorflow_datasets as tfds
from config import *
from preprocess import preprocess_sentence
import argparse


model_version = 1

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="Input a sentence")
args = vars(ap.parse_args())

sentence = args.get("input")
print('Input: {}'.format(sentence))

# 단어 집합(Vocabulary) 불러오기
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(VOCAB_FILE_NAME)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

sentence = preprocess_sentence(sentence)
sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
output = tf.expand_dims(START_TOKEN, 0)


def request_prediction(sentence, output, version):
    # The inputs of model is [inputs, dec_inputs] == [sentence, output]
    # Request a prediction and get the final output
    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        # predictions = model(inputs=[sentence, output], training=False)
        # Serving prediction request
        # sentence and output is Tensor.
        # To use tolist(), numpy() used.
        json_data = json.dumps({"signature_name": "serving_default",
                                "inputs": {"dec_inputs": output.numpy().tolist(), "inputs": sentence.numpy().tolist()}})

        # json_data = json.dumps({"signature_name": "serving_default", "instances": sentence.numpy().tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(f'http://localhost:8501/v1/models/transformer/versions/{version}:predict',
                                      data=json_data,
                                      headers=headers)

        predictions = json.loads(json_response.text)['outputs']
        # list to Tensor
        predictions = tf.convert_to_tensor(predictions)
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    prediction = tf.squeeze(output, axis=0)

    return prediction


prediction = request_prediction(sentence, output, model_version)
print("Prediction: ", prediction)
predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
print('Output: {}'.format(predicted_sentence))


def punctuate(sentence):
    last_character = sentence[-1]
    if sentence.endswith(('?', '.', '!')):
        predicted_sentence = sentence.rstrip(r"?.!")
        s = list(predicted_sentence)
        s[-1] = last_character
        sentence = "".join(s)
    else:
        sentence = sentence + '.'

    return sentence


predicted_sentence = punctuate(predicted_sentence)
print(predicted_sentence)

# TODO: use defined signature
'''
json_data = json.dumps({"signature_name": "serving_default",
                        "inputs": {"dec_inputs": output.numpy().tolist(), "inputs": sentence.numpy().tolist()}})

headers = {"content-type": "application/json"}
json_response = requests.post(f'http://localhost:8501/v1/models/transformer/versions/1:predict',
                              data=json_data,
                              headers=headers)
prediction = json.loads(json_response.text)['outputs']
'''

# --mount type=bind,source=/home/tokim/code/transformer-ko/models/,target=/models
