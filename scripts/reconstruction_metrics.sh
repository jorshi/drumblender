drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_all" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/percussion.yaml
rm logs/config.yaml

drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_a" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a.yaml
rm logs/config.yaml

drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_e" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e.yaml
rm logs/config.yaml
