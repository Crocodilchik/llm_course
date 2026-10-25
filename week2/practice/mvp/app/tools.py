from __future__ import annotations
import json
from pydantic import Field
from .schemas import Strict, clean_text


class CountArgs(Strict):
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    only_24_7: bool = False


class FindArgs(Strict):
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    street: str | None = Field(default=None,min_length=1,max_length=120)
    limit: int = Field(default=5,ge=1,le=10)


class NearestArgs(Strict):
    lat: float = Field(ge=-90,le=90,allow_inf_nan=False)
    lon: float = Field(ge=-180,le=180,allow_inf_nan=False)
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")


class DetailArgs(Strict):
    place_id: int = Field(ge=1)


class BrandArgs(Strict):
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    limit: int = Field(default=5,ge=1,le=50)


class SearchArgs(Strict):
    query: str = Field(min_length=1,max_length=300)
    street: str | None = Field(default=None,min_length=1,max_length=120)
    limit: int = Field(default=5,ge=1,le=10)


ARGUMENTS = dict(count_places=CountArgs,find_places=FindArgs,nearest_place=NearestArgs,
                 place_details=DetailArgs,top_brands=BrandArgs,semantic_search=SearchArgs)


class ToolSet:
    def __init__(self, poi, search, guard, emit, snapshot):
        self.poi,self.search,self.guard,self.emit=poi,search,guard,emit
        self.places = {}
        self.counts = set()
        self.brands = set()
        self.source_failures = 0
        self.snapshot = snapshot
        self.last_ids = list(snapshot.get("place_ids",[]))

    async def before_tool(self, tool, args, tool_context):
        self.guard.before_tool()
        name = tool.name
        try:
            ARGUMENTS[name].model_validate(args)
        except Exception:
            self.guard.rejected(name,args)
            await self.emit("tool_rejected",dict(tool=name,code="invalid_arguments"))
            return dict(error="invalid_arguments",allowed_fields=list(ARGUMENTS[name].model_fields) if name in ARGUMENTS else [])
        return None

    async def execute(self,name,args):
        validated = ARGUMENTS[name].model_validate(args).model_dump()
        await self.emit("tool_started",dict(tool=name))
        try:
            if name=="count_places":
                result=await self.poi.count(**validated)
                self.counts.add(result["count"])
            elif name=="find_places":
                result=dict(places=await self.poi.find(**validated))
            elif name=="nearest_place":
                found=await self.poi.nearest(**validated)
                result=dict(places=[found] if found else [])
            elif name=="place_details":
                found=await self.poi.details(**validated)
                result=dict(places=[found] if found else [])
            elif name=="top_brands":
                result=dict(brands=await self.poi.brands(**validated))
                self.brands.update(r["brand"] for r in result["brands"])
            else:
                validated["query"]=clean_text(validated["query"])
                result=dict(places=await self.search.query(**validated),
                    note="Сходство по названию и тегам OSM; наличие блюда в меню не подтверждено")
            if len(json.dumps(result,ensure_ascii=False))>4000:
                self.guard.rejected(name,args)
                result=dict(error="result_too_large",hint="Уменьшите limit")
            if "places" in result:
                for row in result["places"]:
                    self.places[row["id"]]=row
                if name != "place_details" or not self.last_ids:
                    self.last_ids=[row["id"] for row in result["places"]]
            self.snapshot["last_tool"]=name
            self.snapshot.setdefault("filters",{}).update({k:v for k,v in validated.items() if k not in ("place_id","limit")})
            await self.emit("tool_finished",dict(tool=name,ok="error" not in result,
                count=len(result.get("places",result.get("brands",[])))))
            return result
        except Exception:
            self.source_failures+=1
            if self.source_failures>=2:
                self.guard.stop_code="repeated_source_failure"
            await self.emit("tool_finished",dict(tool=name,ok=False,code="source_unavailable"))
            return dict(error="source_unavailable",hint="Источник недоступен; не выдумывайте результат")

    def functions(self):
        async def count_places(kind:str,only_24_7:bool=False)->dict:
            """Количество мест вида kind; only_24_7=true считает только круглосуточные.

            Args:
                kind: Вид латиницей, например bank, pharmacy, cafe, school.
                only_24_7: Только круглосуточные, иначе все.
            """
            return await self.execute("count_places",dict(kind=kind,only_24_7=only_24_7))
        async def find_places(kind:str,street:str|None=None,limit:int=5)->dict:
            """Список конкретных мест, необязательный фильтр улицы. Не для количества.

            Args:
                kind: Вид места латиницей.
                street: Часть названия улицы или null.
                limit: Число результатов от 1 до 10.
            """
            return await self.execute("find_places",dict(kind=kind,street=street,limit=limit))
        async def nearest_place(lat:float,lon:float,kind:str)->dict:
            """Ближайшее место к координатам; расстояние в метрах, lat=широта, lon=долгота.

            Args:
                lat: Широта в градусах от -90 до 90.
                lon: Долгота в градусах от -180 до 180.
                kind: Вид места латиницей.
            """
            return await self.execute("nearest_place",dict(lat=lat,lon=lon,kind=kind))
        async def place_details(place_id:int)->dict:
            """Карточка известного id из результатов поиска или состояния диалога.

            Args:
                place_id: Положительный id существующего места.
            """
            return await self.execute("place_details",dict(place_id=place_id))
        async def top_brands(kind:str,limit:int=5)->dict:
            """Сети/бренды заданного вида; не список конкретных мест.

            Args:
                kind: Вид места латиницей, например outpost.
                limit: Число брендов от 1 до 50.
            """
            return await self.execute("top_brands",dict(kind=kind,limit=limit))
        async def semantic_search(query:str,street:str|None=None,limit:int=5)->dict:
            """Смысловой поиск заведений питания по названию и тегам кухни OSM.

            Args:
                query: Пожелание, блюдо или описание длиной 1–300 символов.
                street: Необязательный фильтр улицы или null.
                limit: Число результатов от 1 до 10.
            """
            return await self.execute("semantic_search",dict(query=query,street=street,limit=limit))
        return [count_places,find_places,nearest_place,place_details,top_brands,semantic_search]
