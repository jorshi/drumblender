drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_all" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/percussion.yaml
rm logs/config.yaml

drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_a" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a.yaml
rm logs/config.yaml

drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_e" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e.yaml
rm logs/config.yaml
