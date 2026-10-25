from __future__ import annotations
import asyncio
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update
from .models import Run, RunEvent
from .auth import identity

router=APIRouter()
TERMINAL={"completed","failed","cancelled","interrupted"}


class EventWriter:
    def __init__(self,db,run_id):
        self.db,self.run_id=db,run_id

    async def emit(self,kind,payload):
        async with self.db.sessions() as session:
            sequence=await session.scalar(update(Run).where(Run.id==self.run_id).values(
                sequence=Run.sequence+1).returning(Run.sequence))
            if sequence is None:
                return
            session.add(RunEvent(run_id=self.run_id,sequence=sequence,kind=kind,payload=payload))
            if kind=="usage":
                await session.execute(update(Run).where(Run.id==self.run_id).values(**{
                    k:payload[k] for k in ("model_calls","input_tokens","output_tokens")}))
            await session.commit()


async def own_run(request,run_id,user_id):
    async with request.app.state.db.sessions() as session:
        run=await session.get(Run,run_id)
        if not run or run.user_id!=user_id:
            raise HTTPException(404,"run_not_found")
        return run


@router.get("/api/runs/{run_id}/events")
async def stream(run_id:str,request:Request):
    user=await identity(request)
    await own_run(request,run_id,user.id)
    try:
        after=int(request.headers.get("last-event-id",request.query_params.get("after","0")))
        if not 0<=after<=100000:
            raise ValueError()
    except ValueError:
        raise HTTPException(400,"invalid_event_cursor") from None
    async def generate():
        cursor=after
        heartbeat=0
        while not await request.is_disconnected():
            try:
                await identity(request)
            except HTTPException:
                yield 'event: auth_expired\ndata: {"code":"session_expired"}\n\n'
                return
            try:
                async with request.app.state.db.sessions() as session:
                    events=(await session.scalars(select(RunEvent).where(
                        RunEvent.run_id==run_id,RunEvent.sequence>cursor).order_by(RunEvent.sequence))).all()
                    current=await session.get(Run,run_id)
                for event in events:
                    cursor=event.sequence
                    yield f"id: {cursor}\nevent: {event.kind}\ndata: {json.dumps(event.payload,ensure_ascii=False,allow_nan=False)}\n\n"
                if current is None or current.status in TERMINAL:
                    return
            except Exception:
                yield 'event: stream_error\ndata: {"code":"stream_unavailable"}\n\n'
                return
            heartbeat+=1
            if heartbeat>=20:
                yield ": heartbeat\n\n"
                heartbeat=0
            await asyncio.sleep(0.25)
    return StreamingResponse(generate(),media_type="text/event-stream",
        headers={"Cache-Control":"no-cache, no-transform","X-Accel-Buffering":"no"})
