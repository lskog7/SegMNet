# from fastapi import APIRouter, HTTPException, Depends
# from app.schemas.user import UserCreate, UserOut
# from app.services.auth_service import register_user, login_user

# router = APIRouter()

# @router.post("/register", response_model=UserOut)
# async def register(user: UserCreate):
#     return await register_user(user)

# @router.post("/login")
# async def login(user: UserCreate):
#     token = await login_user(user)
#     if not token:
#         raise HTTPException(status_code=400, detail="Invalid credentials")
#     return {"access_token": token}
