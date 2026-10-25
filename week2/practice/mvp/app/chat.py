from __future__ import annotations
import asyncio
import copy
from fastapi import APIRouter,Request,HTTPException
from sqlalchemy import select,func
from .models import Chat,Run,RunEvent
from .schemas import Message,clean_text
from .auth import identity
from .agent import AgentFailure
from .events import EventWriter,own_run,TERMINAL

router=APIRouter()


def run_view(run):
    return dict(id=run.id,chat_id=run.chat_id,prompt=run.prompt,status=run.status,
        answer=run.final_answer,error_code=run.error_code,model_calls=run.model_calls,
        input_tokens=run.input_tokens,output_tokens=run.output_tokens,created_at=run.created_at.isoformat())


class RunManager:
    def __init__(self,db,runtime,settings):
        self.db,self.runtime,self.settings=db,runtime,settings
        self.tasks={}
        self.reservation=asyncio.Lock()
        self.semaphore=asyncio.Semaphore(settings.active_runs)
        self.stopping=False

    async def submit(self,user_id,chat_id,message):
        async with self.reservation:
            async with self.db.sessions() as session:
                chat=await session.get(Chat,chat_id)
                if not chat or chat.user_id!=user_id:
                    raise HTTPException(404,"chat_not_found")
                existing=await session.scalar(select(Run).where(Run.chat_id==chat_id,Run.request_key==message.request_key))
                if existing:
                    return existing
                if await session.scalar(select(func.count()).select_from(Run).where(
                    Run.user_id==user_id,Run.status.in_(["queued","running"]))):
                    raise HTTPException(409,"user_already_running")
                if len(self.tasks)>=5:
                    raise HTTPException(429,"server_capacity")
                prompt=clean_text(message.text.strip())
                if not prompt:
                    raise HTTPException(422,"empty_question")
                run=Run(chat_id=chat_id,user_id=user_id,request_key=message.request_key,prompt=prompt)
                session.add(run)
                if chat.title=="Новый диалог":
                    chat.title=prompt[:60]
                await session.commit()
            await EventWriter(self.db,run.id).emit("queued",dict(status="queued"))
            task=asyncio.create_task(self.execute(run.id))
            self.tasks[run.id]=task
            def finished(done):
                self.tasks.pop(run.id,None)
                if not done.cancelled():
                    done.exception()  # Consume safely; no raw exception in event-loop logs.
            task.add_done_callback(finished)
            return run

    async def finish(self,run_id,status,code=None,answer=None,totals=None,snapshot=None):
        async with self.db.sessions() as session:
            run=await session.get(Run,run_id)
            if not run or run.status in TERMINAL:
                return
            run.status,run.error_code,run.final_answer=status,code,answer
            if totals:
                for key,value in totals.items():
                    setattr(run,key,value)
            if snapshot is not None:
                chat=await session.get(Chat,run.chat_id)
                chat.last_results=list(snapshot.get("place_ids",[]))
                chat.filters=copy.deepcopy(snapshot)
            run.sequence+=1
            session.add(RunEvent(run_id=run.id,sequence=run.sequence,kind=status,payload=dict(status=status,code=code)))
            await session.commit()

    async def execute(self,run_id):
        writer=EventWriter(self.db,run_id)
        try:
            async with self.semaphore:
                async with self.db.sessions() as session:
                    run=await session.get(Run,run_id)
                    if run.status in TERMINAL:
                        return
                    chat=await session.get(Chat,run.chat_id)
                    run.status="running"
                    snapshot=copy.deepcopy(chat.filters)
                    snapshot["place_ids"]=list(chat.last_results)
                    await session.commit()
                await writer.emit("started",dict(status="running"))
                async with asyncio.timeout(self.settings.run_timeout):
                    answer,totals,snapshot=await self.runtime.run(run.user_id,run.chat_id,run.prompt,snapshot,writer.emit)
                await self.finish(run_id,"completed",answer=answer,totals=totals,snapshot=snapshot)
        except asyncio.CancelledError:
            await self.finish(run_id,"interrupted" if self.stopping else "cancelled",
                              "server_stopping" if self.stopping else "user_cancelled")
        except TimeoutError:
            await self.finish(run_id,"failed","run_timeout")
        except AgentFailure as error:
            await self.finish(run_id,"failed",error.code,totals=error.totals)
        except Exception:
            await self.finish(run_id,"failed","internal_error")

    async def cancel(self,run):
        await self.finish(run.id,"cancelled","user_cancelled")
        task=self.tasks.get(run.id)
        if task:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    async def close(self):
        self.stopping=True
        tasks=list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)


@router.get("/api/chats")
async def chats(request:Request):
    user=await identity(request)
    async with request.app.state.db.sessions() as session:
        rows=(await session.scalars(select(Chat).where(Chat.user_id==user.id).order_by(Chat.created_at.desc()).limit(100))).all()
    return [dict(id=c.id,title=c.title) for c in rows]


@router.post("/api/chats",status_code=201)
async def create_chat(request:Request):
    user=await identity(request,csrf=True)
    async with request.app.state.db.sessions() as session:
        chat=Chat(user_id=user.id)
        session.add(chat)
        await session.commit()
    return dict(id=chat.id,title=chat.title)


@router.get("/api/chats/{chat_id}")
async def history(chat_id:str,request:Request):
    user=await identity(request)
    async with request.app.state.db.sessions() as session:
        chat=await session.get(Chat,chat_id)
        if not chat or chat.user_id!=user.id:
            raise HTTPException(404,"chat_not_found")
        runs=(await session.scalars(select(Run).where(Run.chat_id==chat_id).order_by(Run.created_at.desc()).limit(50))).all()
    return dict(id=chat.id,title=chat.title,runs=[run_view(r) for r in reversed(runs)])


@router.post("/api/chats/{chat_id}/messages",status_code=202)
async def send(chat_id:str,body:Message,request:Request):
    user=await identity(request,csrf=True)
    run=await request.app.state.manager.submit(user.id,chat_id,body)
    return dict(id=run.id,status=run.status)


@router.get("/api/runs/{run_id}")
async def get_run(run_id:str,request:Request):
    user=await identity(request)
    return run_view(await own_run(request,run_id,user.id))


@router.post("/api/runs/{run_id}/cancel")
async def cancel(run_id:str,request:Request):
    user=await identity(request,csrf=True)
    run=await own_run(request,run_id,user.id)
    await request.app.state.manager.cancel(run)
    return dict(ok=True)
