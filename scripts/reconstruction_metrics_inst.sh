drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_e_cymbals" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_cymbals.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_e_kick" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_kick.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_e_snare" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_snare.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_e_tom" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_e_tom.yaml
rm logs/config.yaml

drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_a_cymbals" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_cymbals.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_a_kick" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_kick.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_a_snare" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_snare.yaml
rm logs/config.yaml
drumblender test -c eval/$1.yaml --ckpt_path eval/$1.ckpt --trainer.logger CSVLogger --trainer.logger.name $1"_a_tom" --model.test_metrics cfg/metrics/drumblender_metrics.yaml --data cfg/data/filtered/percussion_a_tom.yaml
rm logs/config.yaml
