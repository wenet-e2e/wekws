FRRs with FAR fixed at once per hour:

| model                 | params(K) | epoch     | hi_xiaowen | nihao_wenwen |
|-----------------------|-----------|-----------|------------|--------------|
| GRU                   | 203       | 80(avg30) | 0.088901   | 0.083827     |
| TCN                   | 134       | 80(avg30) | 0.023494   | 0.029884     |
| DS_TCN                | 21        | 80(avg30) | 0.005357   | 0.006390     |
| DS_TCN(spec_aug)      | 21        | 80(avg30) | 0.008176   | 0.005075     |
| MDTC                  | 156       | 80(avg10) | 0.007142   | 0.005920     |
| MDTC_Small            | 31        | 80(avg10) | 0.005357   | 0.005920     |
