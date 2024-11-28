# |--------------------|
# | MAIN DATABASE FILE |
# |--------------------|


from datetime import datetime
from sqlalchemy import Integer, func
from sqlalchemy.orm import DeclarativeBase, declared_attr, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from typing import Annotated
from sqlalchemy import String

from .config import settings


# Define database url from settings:
DATABASE_URL = settings.get_db_url()

# Create async engine:
engine = create_async_engine(url=DATABASE_URL)
# Define async session maker:
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

# Define annotations:
uniq_str_ann = Annotated[str, mapped_column(unique=True)]


# Define base class for all models (tables) in the database:
class Base(AsyncAttrs, DeclarativeBase):
    # Basic attributes:
    __abstract__ = True  # Make it abstract to avoid creation of table

    # Basic columns:
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )  # Obvious ID column
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now()
    )  # TO record creation time
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )  # To record last update time

    # Basic methods:
    #   Automize tablename creation:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + "s"
