pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r examples/requirements.txt --no-build-isolation
# git submodule update --init --recursive
pip install -e . --no-build-isolation


cd examples
python datasets/download_dataset.py