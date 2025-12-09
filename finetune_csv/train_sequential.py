import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

 # 添加项目根目录到系统路径，以便导入项目中的模块
sys.path.append('../')
from model import Kronos, KronosTokenizer, KronosPredictor

# 导入配置加载器和训练相关的函数
from config_loader import CustomFinetuneConfig
from finetune_tokenizer import train_tokenizer, set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, create_dataloaders, setup_logging as setup_basemodel_logging


class SequentialTrainer:
    """
    顺序训练器：负责按顺序执行 Tokenizer（分词器）和 Base Model（基础预测模型）的微调训练。
    量化新手注意：
    1. Tokenizer 负责将连续的时间序列数据离散化为 tokens（类似文本中的词）。
    2. Base Model (Predictor) 负责根据历史 tokens 预测未来的 tokens 或数值。
    """
    
    def __init__(self, config_path: str = None):
        # 加载配置文件
        self.config = CustomFinetuneConfig(config_path)
        
        # 获取分布式训练的环境变量
        # rank: 当前进程的ID（0表示主进程）
        # world_size: 总进程数（通常等于使用的GPU数量）
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.config.device_id if hasattr(self.config, 'device_id') else 0)))
        
        # 设置计算设备（GPU 或 CPU）
        self.device = self._setup_device()
        
        self.config.print_config_summary()
    
    def _setup_device(self):
        """配置计算设备，优先使用 GPU"""
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
        
        if self.rank == 0:
            print(f"Using device: {device} (rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank})")
        return device
    
    def _setup_distributed(self):
        """初始化分布式训练环境 (DDP)，用于多卡并行训练"""
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(f"Distributed training initialized: backend={backend}, world_size={self.world_size}")
        else:
            if self.rank == 0:
                print("Distributed training not enabled, using single GPU/CPU training")
    
    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)
        
        print(f"Tokenizer model exists: {tokenizer_exists}")
        print(f"Basemodel model exists: {basemodel_exists}")
        
        return tokenizer_exists, basemodel_exists
    
    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)
        print(f"Created directory: {self.config.tokenizer_save_path}")
        print(f"Created directory: {self.config.basemodel_save_path}")
    
    def train_tokenizer_phase(self):
        """
        第一阶段：训练 Tokenizer
        目标：学习如何将时间序列数据高效地编码为 tokens。
        
        训练日志解读：
        - VQ Loss (Vector Quantization Loss): 向量量化损失。
          通常包含两部分：
          1. Commit Loss: 确保编码器输出接近量化码本中的向量。
          2. Entropy Loss: 鼓励码本使用的多样性（熵正则化）。
          注意：如果 VQ Loss 为负数，通常是因为其中的 Entropy Loss (熵损失) 项被设计为最大化熵（即最小化负熵），或者使用了某种特定的正则化策略。在某些 VQ-VAE 变体中，为了防止码本坍缩，会减去熵项，导致总 VQ Loss 可能为负。这是正常现象，只要它在训练过程中趋于稳定或收敛即可。
          
        - Recon Loss Pre (Reconstruction Loss Pre): 
          基于前几个关键 bit (s1_bits) 的重构损失。模型试图只用一部分信息粗略地还原原始数据。
          
        - Recon Loss All:
          基于完整 bit (全码本) 的重构损失。模型使用所有信息精确地还原原始数据。
          
        理想情况：随着训练进行，Recon Loss 应该不断下降，代表模型能越来越好地还原原始数据。
        """
        print("\n" + "="*60)
        print("Starting Tokenizer Fine-tuning Phase")
        print("="*60)
        
        # 检查模型是否已存在，避免重复训练
        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer model already exists, skipping training")
            return True
        
        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)
        
        # 设置随机种子，保证实验可复现
        set_seed(self.config.seed)
        
        # 初始化 Tokenizer 模型
        if getattr(self.config, 'pre_trained_tokenizer', True):
            # 情况A：加载预训练好的 Tokenizer（推荐）
            logger.info("Loading pretrained tokenizer...")
            if self.rank == 0:
                print("Loading pretrained tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        else:
            # 情况B：从头开始初始化（随机权重）
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4)
            )
        tokenizer = tokenizer.to(self.device)
        
        model_size = sum(p.numel() for p in tokenizer.parameters())
        logger.info(f"Tokenizer parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Tokenizer parameters: {model_size:,}")
        
        logger.info("=== Training Configuration ===")
        # ... (日志记录)
        
        logger.info("Starting tokenizer fine-tuning training...")
        if self.rank == 0:
            print("Starting tokenizer fine-tuning training...")
        start_time = time.time()
        
        # 调用核心训练函数执行训练
        best_val_loss = train_tokenizer(
            tokenizer,
            self.device,
            self.config,
            self.config.tokenizer_save_path,
            logger,
        )
        training_time = time.time() - start_time
        
        final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nTraining time: {training_time/60:.2f} minutes\nModel saved to: {self.config.tokenizer_save_path}"
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")
        
        return True
    
    def train_basemodel_phase(self):
        """
        第二阶段：训练 Base Model (Predictor)
        目标：利用微调后的 Tokenizer 处理数据，训练预测模型进行未来趋势预测。
        """
        print("\n" + "="*60)
        print("Starting Basemodel Fine-tuning Phase")
        print("="*60)
        
        # 确保第一阶段的 Tokenizer 已经训练完成并保存
        if getattr(self.config, 'pre_trained_tokenizer', True):
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                raise FileNotFoundError(f"Fine-tuned tokenizer does not exist: {self.config.finetuned_tokenizer_path}")
        
        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel model already exists, skipping training")
            return True
        
        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)
        
        set_seed(self.config.seed)
        
        # 加载刚刚微调好的 Tokenizer
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading fine-tuned tokenizer...")
            if self.rank == 0:
                print("Loading fine-tuned tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
        else:
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture for Predictor training")
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4)
            )
        tokenizer = tokenizer.to(self.device)
        
        # 初始化预测模型 (Predictor)
        if getattr(self.config, 'pre_trained_predictor', True):
            logger.info("Loading pretrained predictor...")
            if self.rank == 0:
                print("Loading pretrained predictor...")
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            # 随机初始化预测模型
            if self.rank == 0:
                print("pre_trained_predictor=False, randomly initializing Predictor architecture")
            import json
            cfg_path = os.path.join(self.config.pretrained_predictor_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            print("model_config: ", arch)
            model = Kronos(
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                n_layers=arch.get('n_layers', 12),
                d_model=arch.get('d_model', 832),
                n_heads=arch.get('n_heads', 16),
                ff_dim=arch.get('ff_dim', 2048),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.2),
                token_dropout_p=arch.get('token_dropout_p', 0.0),
                learn_te=arch.get('learn_te', True)
            )
        model = model.to(self.device)
        
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Model parameters: {model_size:,}")
        
        logger.info("=== Training Configuration ===")
        # ... (日志记录)
        
        logger.info("Starting fine-tuning training...")
        if self.rank == 0:
            print("Starting fine-tuning training...")
        start_time = time.time()
        
        # 调用核心训练函数执行训练
        best_val_loss = train_model(
            model,
            tokenizer,
            self.device,
            self.config,
            self.config.basemodel_save_path,
            logger,
        )
        training_time = time.time() - start_time
        
        final_msg = f"Basemodel training completed! Best validation loss: {best_val_loss:.4f}\nTraining time: {training_time/60:.2f} minutes\nModel saved to: {self.config.basemodel_save_path}"
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")
        
        return True
    
    def run_training(self):
        """
        执行完整的训练流程：
        1. 准备目录和环境
        2. 运行 Tokenizer 训练 (如果配置开启)
        3. 运行 Base Model 训练 (如果配置开启)
        """
        if self.rank == 0:
            print("Starting Kronos model sequential fine-tuning training")
            print(f"Experiment name: {self.config.experiment_name}")
            print(f"Experiment description: {self.config.experiment_description}")
        
        self._setup_distributed()
        
        self._create_directories()
        
        tokenizer_exists, basemodel_exists = self._check_existing_models()
        
        total_start_time = time.time()
        
        try:
            if self.config.train_tokenizer:
                success = self.train_tokenizer_phase()
                if not success:
                    print("Tokenizer training failed, terminating training")
                    return False
            else:
                print("Skipping Tokenizer training phase")
            
            if self.config.train_basemodel:
                success = self.train_basemodel_phase()
                if not success:
                    print("Basemodel training failed, terminating training")
                    return False
            else:
                print("Skipping Basemodel training phase")
            
            total_time = time.time() - total_start_time
            
            if self.rank == 0:
                print("\n" + "="*60)
                print("Training completed!")
                print("="*60)
                print(f"Total training time: {total_time/60:.2f} minutes")
                print(f"Tokenizer model: {self.config.tokenizer_best_model_path}")
                print(f"Basemodel model: {self.config.basemodel_best_model_path}")
                print("="*60)
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"Error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            pass


def main():
    parser = argparse.ArgumentParser(description='Kronos Model Sequential Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--skip-tokenizer', action='store_true', 
                       help='Skip tokenizer training phase')
    parser.add_argument('--skip-basemodel', action='store_true', 
                       help='Skip basemodel training phase')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip training for existing models')
    
    args = parser.parse_args()
    
    trainer = SequentialTrainer(args.config)
    
    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True
    
    success = trainer.run_training()
    
    if success:
        print("Training completed successfully!")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    else:
        print("Training failed!")
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
