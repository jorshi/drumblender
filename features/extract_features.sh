# Iteratively runs and collects transient features from all the models and drum types.
# Assumes checkpoints and model configs are stored in a directory ./pretrained.
# Run from drumblender base dir
#   $ source ./features/extract_features.sh

for model in all_parallel noise_parallel_transient_params noise_transient_params transient_params
do
    for instrument in kick snare tom hihat cymbals percussion clap
    do
        for sampletype in a e
        do
            drumblender-film --data ./cfg/data/filtered/percussion_${sampletype}_${instrument}.yaml --split test ./pretrained/${model}.yaml ./pretrained/${model}.ckpt ./features/${model}_percussion_${sampletype}_${instrument}.pt
        done
    done
done
