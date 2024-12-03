# |-----------------------------|
# | DATABASE CONFIGURATION FILE |
# |-----------------------------|


from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# .env file structure for dataset configuration:

# |--------------------------------|
# | [Database Connection Settings] |
# | DB_HOST=000.000.000.000        |
# | DB_PORT=0000                   |
# | DB_NAME=database               |
# | DB_USER=user                   |
# | DB_PASSWORD=password           |
# |--------------------------------|


# Configuration class:
class Settings(BaseSettings):
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env"
    )

    def get_db_url(self):
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()

if __name__ == "__main__":
    print("DB URL =>", settings.get_db_url())
    print("DB HOST =>", settings.DB_HOST)
