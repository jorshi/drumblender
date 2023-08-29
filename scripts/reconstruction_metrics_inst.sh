drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_cymbals" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_cymbals.yaml
rm logs/config.yaml
drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_kick" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_kick.yaml
rm logs/config.yaml
drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_snare" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_snare.yaml
rm logs/config.yaml
drumblender test -c $1/$2.yaml --ckpt_path $1/$2.ckpt --trainer.logger CSVLogger --trainer.logger.name $2"_tom" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_tom.yaml
rm logs/config.yaml
