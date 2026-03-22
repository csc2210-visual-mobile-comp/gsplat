pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r examples/requirements.txt --no-build-isolation
# git submodule update --init --recursive
pip install -e . --no-build-isolation


cd LoRA_modification_and_tests
python datasets/download_dataset.py --dataset refnerf