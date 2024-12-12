from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 内存存储锁信息的结构：{ resource_path: { "locked_by": str, "timestamp": float} }
LOCKS = {}

class LockRequest(BaseModel):
    resource_path: str
    user_id: str

@app.post("/lock")
def lock_resource(lock_req: LockRequest):
    resource = lock_req.resource_path
    user = lock_req.user_id
    if resource in LOCKS:
        # 已锁定
        if LOCKS[resource]["locked_by"] == user:
            # 已由同一用户锁定，可视为续租
            return {"status": "ok", "message": "Already locked by you"}
        else:
            # 被他人锁定
            raise HTTPException(status_code=403, detail=f"Resource locked by {LOCKS[resource]['locked_by']}")
    else:
        # 没有锁，进行锁定
        LOCKS[resource] = {"locked_by": user}
        return {"status": "ok", "message": "Lock acquired"}

@app.post("/unlock")
def unlock_resource(lock_req: LockRequest):
    resource = lock_req.resource_path
    user = lock_req.user_id
    if resource in LOCKS:
        if LOCKS[resource]["locked_by"] == user:
            del LOCKS[resource]
            return {"status": "ok", "message": "Unlocked"}
        else:
            raise HTTPException(status_code=403, detail="Resource locked by another user")
    else:
        # 未锁定则无需解锁
        return {"status": "ok", "message": "Not locked"}

@app.get("/status")
def get_status(resource_path: str):
    if resource_path in LOCKS:
        return {"locked": True, "locked_by": LOCKS[resource_path]["locked_by"]}
    else:
        return {"locked": False, "locked_by": None}


@app.get("/all_locks")
def get_all_locks():
    # LOCKS 格式: { resource_path: {"locked_by": str } }
    all_locks = []
    for resource, info in LOCKS.items():
        all_locks.append({
            "resource_path": resource,
            "locked_by": info["locked_by"]
        })
    return {"locks": all_locks}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
