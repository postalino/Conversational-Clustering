from pathlib import Path
import shutil
import pandas as pd

from config import ROOT_DIR, load_env


class DatasetDownloader:
    def __init__(self, data_dir: Path | None = None, unzip: bool = True) -> None:
        settings = load_env(
            required_keys=["KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_DATASET"]
        )

        self.dataset_name = settings["KAGGLE_DATASET"]
        self.data_dir = data_dir or (ROOT_DIR / "data")
        self.download_dir = self.data_dir / "_tmp_dataset"
        self.csv_path = self.data_dir / "dataset.csv"
        self.unzip = unzip

        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self, force: bool = False) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.csv_path.exists() and not force:
            return self.csv_path

        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)

        self.download_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.api.dataset_download_files(
                dataset=self.dataset_name,
                path=str(self.download_dir),
                unzip=self.unzip,
            )

            csv_files = list(self.download_dir.rglob("*.csv"))

            if len(csv_files) != 1:
                raise ValueError(f"Expected exactly one CSV file, found {len(csv_files)}")

            shutil.copy2(csv_files[0], self.csv_path)

            return self.csv_path

        finally:
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)

    def load_data(
        self,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        df = df[["Id", "Text"]].dropna()

        if sample_size is not None:
            sample_size = min(sample_size, len(df))
            df = df.sample(n=sample_size, random_state=random_state)

        return df.reset_index(drop=True)


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_data()
    data = downloader.load_data(sample_size=100)
    print(data.head())