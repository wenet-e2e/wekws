Comparison among different backbones. FRRs with FAR fixed at once per hour:

| model                 | params(K) | epoch     | hi_xiaowen | nihao_wenwen |
|-----------------------|-----------|-----------|------------|--------------|
| GRU                   | 203       | 80(avg30) | 0.088901   | 0.083827     |
| TCN                   | 134       | 80(avg30) | 0.023494   | 0.029884     |
| DS_TCN                | 287       | 80(avg30) | 0.005357   | 0.006390     |
| DS_TCN(spec_aug)      | 287       | 80(avg30) | 0.008176   | 0.005075     |
| MDTC                  | 156       | 80(avg10) | 0.007142   | 0.005920     |
| MDTC_Small            | 31        | 80(avg10) | 0.005357   | 0.005920     |

Next, we use CTC loss to train the model, with DS_TCN and FSMN. 
and we use CTC prefix beam search to decode and detect keywords, 
the detection is either in non-streaming or streaming fashion. 

Since the FAR is pretty low when using CTC loss, 
the follow result is FRRs with FAR fixed at once per 12 hours:

Comparison between Max-pooling and CTC loss. 
The CTC model is fine-tuned with base model trained on WenetSpeech(23 epoch). 
FRRs with FAR fixed at once per 12 hours
 

| model                 | loss        | hi_xiaowen | nihao_wenwen |
|-----------------------|-------------|------------|--------------|
| DS_TCN(spec_aug)      | Max-pooling | 0.051217   | 0.021896     |
| DS_TCN(spec_aug)      | CTC         | 0.056574   | 0.056856     |


Comparison between DS_TCN(Pretrained with Wenetspeech, 23 epoch) 
and FSMN(modelscope released, xiaoyunxiaoyun model). 
FRRs with FAR fixed at once per 12 hours:

| model                 | params(K)   | hi_xiaowen | nihao_wenwen |
|-----------------------|-------------|------------|--------------|
| DS_TCN(spec_aug)      | 955         | 0.056574   | 0.056856     |
| FSMN(spec_aug)        | 756         | 0.031012   | 0.022460     |

Comparison Between stream_score_ctc and score_ctc. 
FRRs with FAR fixed at once per 12 hours: 

| model                 | stream      | hi_xiaowen | nihao_wenwen |
|-----------------------|-------------|------------|--------------|
| DS_TCN(spec_aug)      | no          | 0.056574   | 0.056856     |
| DS_TCN(spec_aug)      | yes         | 0.132694   | 0.057044     |
| FSMN(spec_aug)        | no          | 0.031012   | 0.022460     |
| FSMN(spec_aug)        | yes         | 0.115215   | 0.020205     |

Note: when using CTC prefix beam search to detect keywords in streaming case(detect in each frame), 
we record the probability of a keyword in a decoding path once the keyword appears in this path.
Actually the probability will increase through the time, so we record a lower value of probability,
which result in a higher False Rejection Rate in Detection Error Tradeoff result. 
The actual FRR will be lower than the DET curve gives in a given threshold. 

Now, the model with CTC loss may not get the best performance, 
but it's more robust compared with the classification model using CE/Max-pooling loss.  
For more result of FSMN-CTC KWS model, you can click [modelscope](https://modelscope.cn/models/damo/speech_charctc_kws_phone-wenwen/summary).