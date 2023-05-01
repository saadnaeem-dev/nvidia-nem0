# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.
````
python transcribe_speech.py \
    model_path=r"C:\Users\saadn\.cache\huggingface\hub\conformer-ctc-fast\stt_en_fastconformer_ctc_large.nemo" \
    pretrained_name=null \
    audio_dir=r"C:\Users\saadn\PycharmProjects\nvidia-riva-hugging-face\MegaAudioC:\Users\saadn\PycharmProjects\nvidia-riva-hugging-face\MegaAudio" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    batch_size=3 \
    compute_timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=True \
    pred_name_postfix="<remove or use another model name for output filename>"
````
--model_path "C:\Users\saadn\.cache\huggingface\hub\conformer-ctc-fast\stt_en_fastconformer_ctc_large.nemo" --audio_dir "C:\Users\saadn\PycharmProjects\nvidia-riva-hugging-face\MegaAudio" --output_filename "test.json" --batch_size 3 --compute_timestamps False --compute_langs False --cuda 0 --amp True --append_pred True --pred_name_postfix "ctc_fast_conformer_"

````
docker run --rm -it --gpus all \
    -v /path/to/audio/input:/input \
    -v /path/to/pretrained/model:/model \
    -v /path/to/output/csv:/output \
    nemo-stt-transcriber --input /input --model_path /model --output /output/transcripts.csv
````

````
docker build --build-arg ARG_NAME=value -t IMAGE_NAME:TAG DIRECTORY
docker build -t nvidia-ngc-nemo --build-arg REQUIRE_AIS_CLI=true 
docker build --build-arg REQUIRE_AIS_CLI=true -t nvidia-ngc-nemo . 
````

````
docker run -it --rm <image_name> /path/to/script arg1 arg2 arg3
docker run -it --rm nvidia-ngc-nemo python transcribe_speech.py --model_path /model --audio_dir /input --output_filename /output/transcripts.csv
````

````
docker run -it --rm <image_name> /path/to/script arg1 arg2 arg3

docker run -it --rm <image_name> /path/to/script arg1 arg2 arg3
````