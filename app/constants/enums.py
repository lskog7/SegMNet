from enum import Enum


class UserRoleEnum(str, Enum):
    ADMIN = "admin"
    USER = "user"


class UserGenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NOT_STATED = "not_stated"
