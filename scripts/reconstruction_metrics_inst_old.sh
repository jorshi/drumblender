drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_clap.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_cymbals.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_hihat.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_kick.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_percussion.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_snare.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_tom.yaml

drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_clap.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_cymbals.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_hihat.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_kick.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_percussion.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_snare.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_tom.yaml
