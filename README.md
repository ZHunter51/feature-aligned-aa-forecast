# Feature-aligned AA-Forecast
Inspired by https://openreview.net/forum?id=TS8HoIWAPQ and AA-Forecast papers. Work-in-progress. All contributions are welcome.
## Data

Data should have form of `data/$SUPERDOMAIN/$DOMAIN.csv`, with three columns:

- `time` denotes the time index.
- `series` denotes the series index.
- `value` denotes the value of the time series at the given time index.

### Source

Data used in the paper is obtained from the following sources:

- [FRED](https://fred.stlouisfed.org)
  - [Commodities](https://fred.stlouisfed.org/categories/32217) category
  - [National Income & Product Accounts](https://fred.stlouisfed.org/categories/18) category
  - [Interest Rates](https://fred.stlouisfed.org/categories/22) category
  - [Exchange Rates](https://fred.stlouisfed.org/categories/15) category
- [NCEI](https://ncei.noaa.gov) (Only 2020s data are used)
  - `"TEMP", "STP", "WDSP", "PRCP"` columns from [Global Surface Summary of the Day - GSOD](https://ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html) dataset
  - `"TAVG", "AWND", "PRCP"` columns from [Global Summary of the Month (GSOM), Version 1.0.3](https://ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00946/html) dataset
  - `"TAVG", "AWND", "PRCP"` columns from [Global Summary of the Year (GSOY), Version 1](https://ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00947/html) dataset

## Usage

```shell
python main.py --source-domains $SOURCE_DOMAIN1 $SOURCE_DOMAIN2 ... \
               --target-domain $TARGET_DOMAIN \
               --forecast-horizon $FORECAST_HORIZON \
               --lookback-multiple $LOOKBACK_MULTIPLE \
               --model $MODEL \
               --loss $LOSS \
               --regularizer $REGULARIZER \
               --temperature $TEMPERATURE \
               --scaler $SCALER \
               --metric $METRIC \
               --learning-rate $LEARNING_RATE \
               --num-lr-cycles $NUM_LR_CYCLES \
               --batch-size $BATCH_SIZE \
               --num-iters $NUM_ITERS \
               --seed $SEED \
               --dtype $DTYPE \
               --data-size $DATA_SIZE
```

The detailed descriptions about the arguments are as follows:
| Argument            | Description                                                                                                                                                                                       | Default      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `source_domains`    | Source domains $\{\mathcal{D}^k\}_k$                                                                                                                                                              |              |
| `target_domain`     | Target domain $\mathcal{D}^T$                                                                                                                                                                     |              |
| `forecast_horizon`  | Forecast horizon $\alpha$                                                                                                                                                                         | `10`         |
| `lookback_multiple` | Lookback multiple $\beta/\alpha$                                                                                                                                                                  | `5`          |
| `model`             | Model architecture $\mathfrak{F}$                                                                                                                                                                 | `"NHiTS"`    |
| `loss`              | Forecasting loss function $\mathcal{L}$                                                                                                                                                           | `"SMAPE"`    |
| `regularizer`       | Regularizer measure $\mathcal{L}_\mathrm{align}$ <br> NOTE: `"None"` for vanilla model                                                                                                            | `"Sinkhorn"` |
| `temperature`       | Regularizing temperature $\lambda$                                                                                                                                                                | `1.0`        |
| `scaler`            | Normalizing function $\sigma$                                                                                                                                                                     | `"softmax"`  |
| `metric`            | Evaluation metric for validation and test                                                                                                                                                         | `"SMAPE"`    |
| `learning_rate`     | Learning rate $\eta$                                                                                                                                                                              | `2e-5`       |
| `num_lr_cycles`     | Number of learning rate cycles<br>NOTE: `torch.optim.lr_scheduler.CyclicLR(mode="triangular2")` is used ([ref](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html)) | `50`         |
| `batch_size`        | Batch size $B$                                                                                                                                                                                    | `2**12`      |
| `num_iters`         | Number of iterations                                                                                                                                                                              | `1000`       |
| `seed`              | Random seed                                                                                                                                                                                       | `0`          |
| `dtype`             | Data type used for `torch` and `numpy`                                                                                                                                                            | `"float32"`  |
| `data_size`         | Fixed data size for each domain <br> NOTE: `"None"` to use all data                                                                                                                               | `75000`      |

