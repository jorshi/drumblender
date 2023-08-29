drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_all --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/percussion.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e_cymbals --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_cymbals.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e_kick --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_kick.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e_snare --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_snare.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_e_tom --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_tom.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a_cymbals --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_cymbals.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a_kick --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_kick.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a_snare --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_snare.yaml
rm logs/config.yaml

drumblender test -c cfg/06_modal.yaml --trainer.logger CSVLogger --trainer.logger.name modal_a_tom --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_tom.yaml
rm logs/config.yaml
