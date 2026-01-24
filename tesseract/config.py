import os
from pathlib import Path

name = Path(__file__).parent.parent.name  # TesserAct (项目根目录名)

tos_root = Path(os.environ.get("TOS_ROOT", "/root/tos"))
vepfs_root = Path(os.environ.get("VEPFS_ROOT", "/vepfs"))

class PathConfig:    
    def __init__(self, exp_name: str):
        self.tos_root = Path(tos_root) if tos_root else Path(globals()["tos_root"])
        self.vepfs_root = Path(vepfs_root) if vepfs_root else Path(globals()["vepfs_root"])
        self.raw_data_dir = self.tos_root / "datasets"
        self.data_dir = self.vepfs_root / "datasets"
        self.processed_data_dir = self.vepfs_root / "datasets-processed"

        self.pwd = self.vepfs_root / name
        self.output_dir = self.pwd / "output"
        self.exp_dir = self.output_dir / exp_name

    def get_raw_dataset_path(self, dataset_name: str) -> Path:
        return self.raw_data_dir / dataset_name
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        return self.data_dir / dataset_name

    def get_processed_dataset_path(self, dataset_name: str = "bridge") -> Path:
        return self.processed_data_dir / dataset_name
    
    def get_results_dir(self) -> Path:
        return self.output_dir / "results"

    def get_logs_dir(self) -> Path:
        return self.exp_dir / "logs"
    
    def __repr__(self):
        return f"PathConfig(tos={self.tos_root}, vepfs={self.vepfs_root})"

