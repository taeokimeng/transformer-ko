# Transformer for Korean Chatbot

---
Transformer for Korean Chatbot has been trained [chatbot data](https://github.com/songys/Chatbot_data).

## Load the repository and the model

1. Clone the repository
~~~
git clone https://github.com/taeokimeng/transformer-ko.git
~~~

2. Load the model

* If you haven't installed git lfs, please install git lfs first.
~~~
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
~~~
* If you already have installed git lfs, pull the real data.
~~~
git lfs install
git lfs pull
~~~

## Use the chatbot with TensorFlow Serving

1. Run TensorFlow Serving (Change the directory to transformer-ko before running)

~~~
./serving_run.sh
~~~

2. Run serving.py and input a sentence

~~~
python serving.py -i "어떤 영화 볼까요?"
~~~

3. Check the answer from the chatbot

~~~
Input: 어떤 영화 볼까요?
Output: 최신 영화가 좋을 것 같아요.
~~~