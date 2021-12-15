FRRs with FAR fixed at once per hour:

| model                 | params(K) | epoch     | hi_xiaowen | nihao_wenwen |
|-----------------------|-----------|-----------|------------|--------------|
| GRU                   | 203       | 80(avg30) | 0.088901   | 0.083827     |
| TCN                   | 134       | 80(avg30) | 0.023494   | 0.029884     |
| DS_TCN                | 21        | 80(avg30) | 0.019641   | 0.018325     |
| DS_TCN(big)           | 287       | 80(avg30) | 0.012217   | 0.011653     |
| DS_TCN(big, quantize) | 287       | 80(avg30) | 0.017292   | 0.022178     |
| MDTC                  | 156       | 80(avg10) | 0.007142   | 0.005920     |
| MDTC_Small            | 31        | 80(avg10) | 0.005357   | 0.005920     |
