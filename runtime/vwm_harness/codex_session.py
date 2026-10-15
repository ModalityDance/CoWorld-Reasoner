"""Persistent goals and native tools through the installed Codex CLI."""
from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Any


@dataclass
class ToolResponse:
    """Ordered text and image evidence from an experiment tool."""

    content_items: list[dict]


def require_codex() -> str:
    executable = shutil.which(os.environ.get("COEVOLVE_CODEX_BIN") or "codex")
    if not executable:
        raise RuntimeError("codex_cli_required:install_Codex_or_set_COEVOLVE_CODEX_BIN")
    if not shutil.which("git"):
        raise RuntimeError("git_required_for_project_workspace")
    return executable


class CodexSession:
    """One App Server connection; Codex owns reasoning, tools and continuation."""

    def __init__(self, workspace: Path, output: Path, *, model: str,
                 base_url: str, api_key: str, timeout: float = 900,
                 writable_roots: tuple[Path, ...] = ()):
        self.workspace, self.output = workspace.resolve(), output.resolve()
        self.timeout = timeout
        self.output.mkdir(parents=True, exist_ok=True)
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.pending: dict[int, queue.Queue] = {}
        self.lock = threading.Lock()
        self.request_id = 0
        self.received_sequence = 0
        self.response_sequence = 0
        self.goal_sequence_floor = 0
        self.thread_id: str | None = None
        self.goal: dict | None = None
        self.active_turns: set[str] = set()
        self.closed = False
        provider = {"name": "CoWorld", "base_url": base_url,
                    "env_key": "AGENTIC_VLM_API_KEY", "wire_api": "responses",
                    "requires_openai_auth": False, "request_max_retries": 2,
                    "stream_max_retries": 2, "stream_idle_timeout_ms": int(timeout * 1000)}
        provider_toml = "{" + ",".join(k + "=" + json.dumps(v) for k, v in provider.items()) + "}"
        settings = {
            "model": json.dumps(model), "model_provider": '"coworld"',
            "model_providers.coworld": provider_toml,
            "model_reasoning_effort": '"medium"', "features.goals": "true",
            "features.multi_agent": "false", "features.plugins": "false",
            "features.apps": "false", "features.hooks": "false",
            "features.skip_host_skill_discovery": "true", "project_doc_max_bytes": "32768",
            "sandbox_workspace_write.writable_roots": json.dumps(
                [str(path.resolve()) for path in writable_roots]),
            "web_search": '"disabled"', "sandbox_workspace_write.network_access": "true",
            "shell_environment_policy.exclude": '["*KEY*","*TOKEN*","*SECRET*","*PASSWORD*"]',
        }
        if os.name == "nt":
            settings["windows.sandbox"] = '"unelevated"'
        command = [require_codex(), "-a", "never", "-C", str(self.workspace), "-s", "workspace-write"]
        for key, value in settings.items():
            command.extend(["-c", key + "=" + value])
        command.extend(["app-server", "--stdio"])
        self.log = (self.output / "events.jsonl").open("a", encoding="utf-8")
        self.stderr = (self.output / "stderr.log").open("a", encoding="utf-8")
        native_home = self.output / "home"
        native_home.mkdir(exist_ok=True)
        environment = {**os.environ, "AGENTIC_VLM_API_KEY": api_key, "CODEX_HOME": str(native_home)}
        for key in ("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", "CODEX_THREAD_ID"):
            environment.pop(key, None)
        self.process = subprocess.Popen(command, cwd=self.workspace,
            env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding="utf-8", bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.metadata = self.request("initialize", {
                "clientInfo": {"name": "coworld_supervisor", "version": "1.0"},
                "capabilities": {"experimentalApi": True, "requestAttestation": False}})
            self._send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def _send(self, message: dict) -> None:
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("codex_process_exited")
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                self.log.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.log.flush()
                self.received_sequence += 1
                message["_received_sequence"] = self.received_sequence
                if "id" in message and "method" not in message:
                    waiter = self.pending.get(message["id"])
                    if waiter:
                        waiter.put(message)
                else:
                    self.events.put(message)
        finally:
            self.events.put(None)
            for waiter in list(self.pending.values()):
                waiter.put({"error": {"message": "codex_connection_closed"}})

    def request(self, method: str, params: dict, timeout: float = 30) -> dict:
        self.request_id += 1
        request_id = self.request_id
        waiter: queue.Queue = queue.Queue()
        self.pending[request_id] = waiter
        try:
            self._send({"id": request_id, "method": method, "params": params})
            response = waiter.get(timeout=timeout)
            self.response_sequence = response.get("_received_sequence", 0)
            if "error" in response:
                raise RuntimeError(f"codex_rpc:{method}:{response['error']}")
            return response.get("result", {})
        except queue.Empty as exc:
            raise TimeoutError(f"codex_rpc_timeout:{method}") from exc
        finally:
            self.pending.pop(request_id, None)

    def start(self, instructions: str, tools: list[dict], *, thread_id: str | None = None) -> str:
        params = {"cwd": str(self.workspace), "approvalPolicy": "never",
                  "sandbox": "workspace-write", "developerInstructions": instructions}
        if thread_id:
            params.update(threadId=thread_id, excludeTurns=True)
            response = self.request("thread/resume", params)
        else:
            params.update(dynamicTools=tools)
            response = self.request("thread/start", params)
        self.thread_id = response["thread"]["id"]
        return self.thread_id

    def set_goal(self, objective: str | None = None, *, status: str = "active") -> dict:
        params = {"threadId": self.thread_id, "status": status}
        if objective is not None:
            params["objective"] = objective
        self.goal = self.request("thread/goal/set", params)["goal"]
        self.goal_sequence_floor = self.response_sequence
        return self.goal

    def get_goal(self) -> dict | None:
        self.goal = self.request("thread/goal/get", {"threadId": self.thread_id})["goal"]
        return self.goal

    def _tool_content(self, result: Any, call: dict) -> list[dict]:
        if not isinstance(result, ToolResponse):
            return [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False)}]
        content = []
        attachments = [{"type": "text", "text":
            f"Visual evidence from {call['tool']} (call {call['callId']}): "
            + json.dumps(call.get("arguments", {}), ensure_ascii=False)}]
        label = ""
        image_count = 0
        for item in result.content_items:
            if item["type"] == "inputText":
                label = item["text"]
                content.append(item)
            elif item["type"] == "inputImage":
                image_count += 1
                attachments.extend([{"type": "text", "text": label},
                                    {"type": "image", "url": item["imageUrl"]}])
                content.append({"type": "inputText", "text": f"Image attachment {image_count}: {label}"})
            else:
                raise ValueError(f"unsupported_tool_content:{item['type']}")
        if image_count:
            # The script bridge serializes dynamic-tool results. Deliver pixels through
            # the native active-turn image input before releasing the tool response.
            self.request("turn/steer", {"threadId": self.thread_id,
                "expectedTurnId": call["turnId"], "input": attachments})
        return content

    def follow_goal(self, tool_handler: Callable[[str, dict], Any],
                    on_event: Callable[[dict], None] | None = None) -> dict:
        """Observe the native continuation returned by Codex."""
        last_progress = time.monotonic()
        while True:
            try:
                event = self.events.get(timeout=10)
            except queue.Empty:
                if time.monotonic() - last_progress > self.timeout:
                    raise TimeoutError("codex_goal_idle_timeout")
                continue
            if event is None:
                raise RuntimeError("codex_connection_closed_before_goal_completion")
            last_progress = time.monotonic()
            method, params = event.get("method"), event.get("params", {})
            if method == "turn/started":
                self.active_turns.add(params["turn"]["id"])
            elif method == "turn/completed":
                self.active_turns.discard(params["turn"]["id"])
                if params["turn"].get("status") == "failed":
                    raise RuntimeError(f"codex_turn_failed:{params['turn'].get('error')}")
            elif method == "thread/goal/updated":
                if event.get("_received_sequence", 0) > self.goal_sequence_floor:
                    self.goal = params["goal"]
            elif method == "item/tool/call" and "id" in event:
                try:
                    result = tool_handler(params["tool"], params.get("arguments") or {})
                    content = self._tool_content(result, params)
                    success = True
                except Exception as exc:
                    result, success = {"error": f"{type(exc).__name__}:{exc}"}, False
                    content = [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False)}]
                self._send({"id": event["id"], "result": {
                    "contentItems": content,
                    "success": success}})
                last_progress = time.monotonic()
            elif "id" in event:
                self._send({"id": event["id"], "error": {
                    "code": -32601, "message": f"unsupported_server_request:{method}"}})
            if on_event:
                on_event(event)
            if self.goal and self.goal.get("status") != "active" and not self.active_turns:
                return self.goal

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            if self.thread_id and self.goal and self.goal.get("status") == "active":
                try:
                    self.request("thread/goal/set", {"threadId": self.thread_id, "status": "paused"})
                except Exception:
                    pass
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=2)
        self.log.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
