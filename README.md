# Transformer for Korean Chatbot

---
Transformer for Korean Chatbot has been trained [chatbot data](https://github.com/songys/Chatbot_data).

## Use the chatbot with TensorFlow Serving

1. Run TensorFlow Serving

``` serving_run.sh ```

2. Run serving.py and input a sentence

``` python serving.py -i "어떤 영화 볼까요?" ```

3. Check the answer from the chatbot

~~~
Input: 어떤 영화 볼까요?
Output: 최신 영화가 좋을 것 같아요.
~~~