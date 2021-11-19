# wenet-kws

Production First and Production Ready End-to-End Keyword Spotting Toolkit.

The goal of this toolkit it to...

Small footprint keyword spotting (KWS), or specifically wake-up word (WuW) detection is a typical and important module in internet of things (IoT) devices.  It provides a way for users to control IoT devices with a hands-free experience. A WuW detection system usually runs locally and persistently on IoT devices, which requires low consumptional power, less model parameters, low computational comlexity and to detect predefined keyword in a streaming way, i.e., requires low latency.


## Typical Scenario

We are going to support the following typical applications of wakeup word:

* Single wake-up word
* Multiple wake-up words
* Customizable wake-up word
* Personalized wake-up word, i.e. combination of wake-up word detection and voiceprint

## Installation

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Dataset

We plan to support a variaty of open source wake-up word datasets, include but not limited to:

* [Hey Snips](https://github.com/sonos/keyword-spotting-research-datasets)
* [Google Speech Command](https://arxiv.org/pdf/1804.03209.pdf)
* [Hi Miya(你好米雅)](http://www.aishelltech.com/wakeup_data)
* [Hi Xiaowen(你好小问)](http://openslr.org/87/)

All the well-trained models on these dataset will be made public avaliable.


## Runtime

We plan to support a variaty of hardwares and platforms, including:

* Web browser
* x86
* Android
* Raspberry Pi

