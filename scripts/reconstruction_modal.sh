drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_all --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/percussion.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_cymbals --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_cymbals.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_kick --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_kick.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_snare --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_snare.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_tom --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_tom.yaml
rm logs/config.yaml
