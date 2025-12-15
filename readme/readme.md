uv venv
source .venv/bin/activate
uv pip install pyqlib -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


git config --global user.name "luyangcai"
git config --global user.email lyc557@163.com


torchrun --nproc_per_node=1 train_tokenizer.py
torchrun --nproc_per_node=1 train_predictor.py