from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    HOST: str
    PORT: int

    GROQ_API_KEY: str
    DB_FIX_AGENT_URL: str

    class Config:
        env_file = ".env"


settings = Settings()