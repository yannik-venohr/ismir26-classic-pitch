# ISMIR 2026: ClassicPitch

This repository contains the code for the paper:

> Yannik Venohr and Christof Weiß. Robust Instrument-Agnostic Music Transcription for Western Classical Music. In *Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)*, 2026.

To transcribe audio with the (latest) ClassicPitch models, use the [ClassicPitch inference repository](https://github.com/yannik-venohr/classic-pitch) instead.

If you have any questions, feel free to reach out to me: `yannik.venohr@uni-wuerzburg.de`.

## Figures and tables

[`ismir2026.ipynb`](ismir2026.ipynb) reproduces the tables and figures of the paper from the track-wise evaluation results of our runs in [`results/`](results/).

## Experiments

All experiments of the paper are in [`configs/experiment/`](configs/experiment/):

| Experiment | Paper | Description |
| --- | --- | --- |
| `CP-XS`, `CP-S`, `CP-M`, `CP-L`, `CP-XL` | Tables 1, 5 | ClassicPitch in five model sizes; `CP-L` is "CP" in the paper and the default |
| `BP+` | Table 5 | Basic Pitch architecture, trained with the data and loss of CP |
| `DRCNN-M+` | Table 4 | DRCNN re-implementation, trained with the data of CP, frame loss only |
| `CP-frame_loss` | Table 4 (CP-Lf) | Frame loss only |
| `CP-onset_loss` | Table 5 (CP-Lo) | Onset loss only |
| `L1-BCE` | Table 5 | Onset loss without shift tolerance and positive weighting |
| `L2-BCE+tolerance` | Table 5 | Onset loss with shift tolerance, without positive weighting |
| `L3-BCE+weight` | Table 5 | Onset loss with positive weighting, without shift tolerance |
| `CP-MusicNet` | Table 5 | Trained on MusicNet only |
| `CP-Less` | Table 5 | Trained on 25% of the training data |
| `R1-Overfit` | Table 5 | All test sets included in the training data |
| `CP-MAESTRO` | Section 5.3 | Trained and tested on the MAESTRO v3 splits |
| `CP-eval-MAESTRO` | Section 5.3 | A trained CP-L evaluated on the MAESTRO v3 test set (needs `checkpoint_eval=<path>`) |

CP-L was trained with the seeds 42, 43 and 44; Table 5 reports their mean and standard deviation. All other experiments use seed 42.

## Data

[`configs/data/ismir2026_splits.csv`](configs/data/ismir2026_splits.csv) lists every track with its dataset and one boolean column per split:

| Column | Meaning |
| --- | --- |
| `train`, `val`, `test` | Main splits of the paper |
| `train25` | 25% subset of the training data (`CP-Less`) |
| `musicnet` | MusicNet (not part of the `train` column) |
| `maestro_train`, `maestro_val`, `maestro_test` | Official MAESTRO v3 splits |

The datasets are not included and must be downloaded separately. We did some minor preprocessing (such as flattening the dirs) but tried to keep them as close to the original as possible. All loading logic is in [`src/data/loaders.py`](src/data/loaders.py). Once you have them prepared, you can edit [`configs/environment/default.yaml`](configs/environment/default.yaml) and enter the audio and annotation directories of the datasets you have. Tracks of datasets that are missing or have no entry there are skipped.


On first use, the HCQT input features and the annotations of every track are computed and saved to `data.feature_dir` (default `./features`), in a subfolder named after a hash of the feature parameters. Later runs reuse them. On a cluster, set `data.node_cache_dir` to a fast local disk so the features are copied there before training.

## Structure

```
train.py                    # entry point: training and evaluation
evaluate_note_level_baselines.py  # evaluation of the MIDI output of all models in Table 5
configs/
  config.yaml               # top-level config
  data/                     # features, splits, dataloader
  model/                    # architectures and decoding thresholds
  training/                 # loss and optimization
  experiment/               # the experiments of the paper
  environment/              # dataset paths, output dir, device
src/
  classic_pitch.py          # LightningModule: loss, optimization, full-track prediction
  loss.py                   # frame loss and shift-tolerant onset loss
  model/                    # HCQT encoder, onset module, baselines
  data/                     # feature computation, dataset loading, patching
  postprocess/              # decoding of activations into note events
  eval/                     # test-set prediction and metrics
results/                    # evaluation results of the paper's runs
ismir2026.ipynb             # tables and figures of the paper
```
## Training and evaluation

### Setup

Requires Python 3.10.

```
pip install -r requirements.txt
```
### Train
```
python train.py experiment=CP-L
```

This trains the model, then evaluates the best checkpoint (lowest validation loss) on the test set. Results go to `output_dir/<experiment_name>_<seed>/`:

```
best_model.ckpt, last.ckpt
logs/                          # CSV logs, track_wise_evaluation.csv
predictions/                   # one MIDI file per test track
hydra/                         # the resolved config of the run
```

Any config value can be overridden on the command line, for example `seed=1`, `device=mps training.precision=32-true` (Apple silicon), or `logging.logger=[csv,wandb]` (Weights & Biases logging; requires `pip install wandb`).

| Option | Effect |
| --- | --- |
| `train=resume` | Continue an interrupted run from its `last.ckpt` |
| `train=False` | Evaluate only, using `run_dir/best_model.ckpt` or `checkpoint_eval=<path>` |
| `recompute_features=True` | Recompute the precomputed features |
| `save_frame_predictions=True` | Also save the frame-wise note and onset activations |

Checkpoints of the ClassicPitch models (not the baselines) can be loaded directly by the inference repository.

## Credits

Parts of the code are adapted from, or inspired by:

- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template): project structure with Lightning and Hydra
- [christofw/multipitch_architectures](https://github.com/christofw/multipitch_architectures): DRCNN baseline (`src/model/mpe.py`)
- [gudgud96/basic-pitch-torch](https://github.com/gudgud96/basic-pitch-torch): Basic Pitch baseline (`src/model/basic_pitch.py`)
- [geoffroypeeters/deeplearning-101-audiomir_notebook](https://github.com/geoffroypeeters/deeplearning-101-audiomir_notebook): ConvNeXt building blocks (`src/model/building_blocks.py`)


## License

[CC BY-NC 4.0](LICENSE): free for non-commercial use with attribution.
