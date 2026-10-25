from __future__ import annotations
import hashlib
import json


class RunStopped(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RunGuard:
    def __init__(self):
        self.model_calls = 0
        self.tool_calls = 0
        self.bad = {}
        self.stop_code = None
        self.usage = {}

    def before_model(self):
        if self.stop_code:
            raise RunStopped(self.stop_code)
        if self.model_calls >= 6:
            self.stop_code = "model_call_limit"
            raise RunStopped("model_call_limit")
        self.model_calls += 1
        self.usage[self.model_calls] = None

    def before_tool(self):
        self.tool_calls += 1
        if self.tool_calls > 20:
            self.stop_code = "tool_call_limit"
            raise RunStopped(self.stop_code)

    def rejected(self, name, args):
        key = hashlib.sha256(json.dumps([name,args],sort_keys=True,default=str).encode()).hexdigest()
        self.bad[key] = self.bad.get(key,0) + 1
        if self.bad[key] >= 2:
            self.stop_code = "repeated_invalid_arguments"

    def record_usage(self, metadata):
        if metadata is None:
            return
        ins = getattr(metadata,"prompt_token_count",None)
        outs = getattr(metadata,"candidates_token_count",None)
        if type(ins) is int and type(outs) is int and ins >= 0 and outs >= 0:
            previous = self.usage.get(self.model_calls)
            self.usage[self.model_calls] = (max(ins,previous[0]),max(outs,previous[1])) if previous else (ins,outs)

    def totals(self):
        complete = bool(self.usage) and all(v is not None for v in self.usage.values())
        return dict(model_calls=self.model_calls,
                    input_tokens=sum(v[0] for v in self.usage.values()) if complete else None,
                    output_tokens=sum(v[1] for v in self.usage.values()) if complete else None)
