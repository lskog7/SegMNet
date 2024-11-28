from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import text

from app.database import Base, uniq_str_ann
from app.constants import UserGenderEnum, UserRoleEnum


# Define Users table as a class:
class User(Base):

    username: Mapped[uniq_str_ann]
    email: Mapped[uniq_str_ann]
    password: Mapped[str]
    gender: Mapped[UserGenderEnum] = mapped_column(
        default=UserGenderEnum.NOT_STATED, server_default=text("'NOT_STATED'")
    )
    role: Mapped[UserRoleEnum] = mapped_column(
        default=UserRoleEnum.USER, server_default=text("'USER'")
    )
