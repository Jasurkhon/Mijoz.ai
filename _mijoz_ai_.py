
import sys
! sudo apt-get install git-lfs

! git clone https://github.com/mozilla/DeepSpeech --branch v0.7.4

! git lfs pull

! pip3 install virtualenv
! virtualenv -p python3 $HOME/tmp/deepspeech-train-venv/

! source $HOME/tmp/deepspeech-train-venv/bin/activate

# %cd /content/DeepSpeech
! pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3

# %cd /content/DeepSpeech
! pip3 install --upgrade -e .

! pip3 uninstall tensorflow
! pip3 install 'tensorflow-gpu==1.15.4'

# %cd /content
# %mkdir uz
# %cd uz
! wget https://drive.google.com/uc?id=1-n31GKOjMyMvcz5RtfX6gOmERjYlBBAT&export=download

# %cd /content/uz
! tar xvzf uc?id=1-n31GKOjMyMvcz5RtfX6gOmERjYlBBAT

! sudo apt-get install sox libsox-fmt-mp3

# %cd /content/DeepSpeech/
! bin/import_cv2.py ../uz/cv-corpus-7.0-2021-07-21/uz

! python3 training/deepspeech_training/util/check_characters.py -alpha -unicode -csv ../uz/cv-corpus-7.0-2021-07-21/uz/clips/train.csv,../uz/cv-corpus-7.0-2021-07-21/uz/clips/dev.csv,../uz/cv-corpus-7.0-2021-07-21/uz/clips/test.csv >> /content/uz/alphabet.txt

# %cd /content/DeepSpeech/data/lm
! wget https://drive.google.com/uc?id=1j3T95y11bUm5oltT800K-OqfUfs8DINm&export=download
! tar -xzvf uc?id=1j3T95y11bUm5oltT800K-OqfUfs8DINm

# %cd /content
!wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
# %mkdir -p /content/kenlm/build

# %cd /content/kenlm/build
! cmake ..

! make -j 4

# %cd /content/DeepSpeech/data/lm
! python3 generate_lm.py --input_txt uzb2.uz-en.uz --output_dir . \
  --top_k 500000 --kenlm_bins /content/kenlm/build/bin/ \
  --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
  --binary_a_bits 255 --binary_q_bits 8 --binary_type trie --discount_fallback

# %cd /content/DeepSpeech/data/lm/
! python3 ./generate_package.py --alphabet /content/uz/alphabet.txt --lm lm.binary --vocab vocab-500000.txt \
  --package kenlm.scorer --default_alpha 0.931289039105002 --default_beta 1.1834137581510284

# %cd /content/DeepSpeech/data/lm/
! python3 ../../lm_optimizer.py \
  --test_files /content/uz/clips/validated.csv \
  --checkpoint_dir /content/checkpoint-lm \
  --load_evaluate init \
  --scorer kenlm.scorer \
  --alphabet_config_path /content/uz/alphabet.txt

# %cd /content/DeepSpeech/
! python3 DeepSpeech.py \
  --train_files /content/uz/cv-corpus-7.0-2021-07-21/uz/clips/train.csv \
  --dev_files /content/uz/cv-corpus-7.0-2021-07-21/uz/clips/dev.csv \
  --test_files /content/uz/cv-corpus-7.0-2021-07-21/uz/clips/test.csv \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 100 \
  --checkpoint_dir ../checkpoint \
  --export_dir ../model \
  --alphabet_config_path ../uz/alphabet.txt \
  --scorer data/lm/kenlm.scorer

# %cd /content/DeepSpeech/
! python3 util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target .

! ./convert_graphdef_memmapped_format --in_graph=/content/model/output_graph.pb --out_graph=/content/model/output_graph.pbmm

! pip3 install deepspeech-gpu

! deepspeech --model /content/model/output_graph.pbmm --audio /content/uz/cv-corpus-7.0-2021-07-21/uz/clips/common_voice_uz_27043256.wav --scorer /content/DeepSpeech/data/lm/kenlm.scorer

# %cd /content
! pip install youtube-dl
yt="https://www.youtube.com/watch?v=CIUfdwISv4c"
! youtube-dl --extract-audio --audio-format wav {yt}

from IPython.display import Audio
Audio("/content/temp.wav")

! deepspeech --model /content/model/output_graph.pbmm --audio /content/temp.wav --scorer /content/DeepSpeech/data/lm/kenlm.scorer