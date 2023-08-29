# Reconstruction Metrics

Run the following commands from the project root to compute reconstruction metrics that were reported in the paper.

Requires:
- Dataset (unfortunately, it is private)
- Pre-trained models, which can downloaded by running the following command in the project root:
    ```bash
    git submodule update --init --recursive
    ```

## Commands to run test metrics
### Cross intrument metrics:

```bash
./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 all_parallel && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_parallel_transient_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_transient_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 transient_params
```

### Instrument dependent metrics:
```bash
./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 all_parallel && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_parallel_transient_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_transient_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 transient_params
```

### Modal-only metrics
```bash
./scripts/reconstruction_modal.sh
```

## Compile Results
After running all the reconstruction evaluation scripts, results will be saved in a series of csv files in a logs directory. To compile these results into Latex tables used in the paper, these scripts can be used:

```bash
python scripts/compile_results.py logs all
python scripts/compile_results.py logs instrument
```
