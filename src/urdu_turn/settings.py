from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str = ""
    data_dir: Path = Path("data")

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def hf_cache(self) -> Path:
        return self.data_dir / "hf"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"
