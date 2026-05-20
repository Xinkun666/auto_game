import argparse
import importlib
import os
import sys
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

try:
    from .protocol import decode_base64_to_frame_rgb, json_dumps, json_loads, to_jsonable
except ImportError:
    from protocol import decode_base64_to_frame_rgb, json_dumps, json_loads, to_jsonable


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass
class SessionState:
    session_id: str
    project_case: str
    game_case: str
    current_stage: Optional[str] = None
    screen: Optional[Tuple[int, int]] = None
    stage_dict: Dict[str, bool] = field(default_factory=dict)
    frame_index: int = 0
    stopped: bool = False


class RemoteFrameWorker:
    def __init__(self, session: SessionState, frame, stage_info: Dict[str, Any]):
        self.session = session
        self.frame = frame
        self.stage_info = stage_info or {}
        self.stage_dict = session.stage_dict
        self.current_stage = session.current_stage
        self.actions = []
        self.finished = False
        self.running = True
        self.failed = False

    def _record(self, action_type: str, *args, **kwargs):
        self.actions.append(
            {
                "type": action_type,
                "args": to_jsonable(list(args)),
                "kwargs": to_jsonable(kwargs),
            }
        )

    def get_info(self, area_name):
        suffix = f"__{area_name}"
        for key, value in self.stage_info.items():
            if key.endswith(suffix):
                return value
        return None

    def get_stage(self):
        return self.current_stage

    def change_stage(self, stage_name):
        old_stage = self.current_stage
        if self.stage_dict and stage_name not in self.stage_dict:
            print(f"[RemoteBrain] stage '{stage_name}' not in STAGE_DICT")
            return
        for key in self.stage_dict.keys():
            self.stage_dict[key] = False
        if self.stage_dict:
            self.stage_dict[stage_name] = True
        self.current_stage = stage_name
        self.session.current_stage = stage_name
        print(f"[RemoteBrain] STATUS CHANGE: [{old_stage}] -> [{stage_name}]")
        self._record("change_stage", stage_name)

    def refresh_frame(self):
        self._record("refresh_frame")
        return False

    def stop(self):
        self.session.stopped = True
        self.running = False
        self.finished = True
        self._record("stop")

    def click(self, *args, **kwargs):
        self._record("click", *args, **kwargs)

    def click_down(self, *args, **kwargs):
        self._record("click_down", *args, **kwargs)

    def tap_single(self, *args, **kwargs):
        self._record("tap_single", *args, **kwargs)

    def tap_double(self, *args, **kwargs):
        self._record("tap_double", *args, **kwargs)

    def move_press(self, *args, **kwargs):
        self._record("move_press", *args, **kwargs)

    def move_to(self, *args, **kwargs):
        self._record("move_to", *args, **kwargs)

    def move_up(self, *args, **kwargs):
        self._record("move_up", *args, **kwargs)

    def mark_failed(self, code, reason, **details):
        self.failed = True
        self._record("stop")
        print(f"[RemoteBrain] mark_failed: code={code}, reason={reason}, details={details}")
        return True


class RemoteBrainCore:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.lock = threading.RLock()
        self.loaded_key = None
        self.stage_controller = None
        self.logic_module = None
        self.info_module = None

    def start_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        project_case = payload.get("project_case") or "Auto_PUBG_ALL"
        game_case = payload.get("game_case") or "auto_pubg"
        session_id = payload.get("session_id") or "default"
        screen = self._parse_screen(payload.get("screen"))

        with self.lock:
            self._ensure_loaded(project_case, game_case, screen)
            stage_dict = dict(getattr(self.info_module, "STAGE_DICT"))
            current_stage = payload.get("current_stage") or self._first_active_stage(stage_dict)
            if current_stage in stage_dict:
                for key in stage_dict.keys():
                    stage_dict[key] = False
                stage_dict[current_stage] = True

            session = SessionState(
                session_id=session_id,
                project_case=project_case,
                game_case=game_case,
                current_stage=current_stage,
                screen=screen,
                stage_dict=stage_dict,
            )
            self.sessions[session_id] = session

        return {
            "ok": True,
            "session_id": session_id,
            "current_stage": current_stage,
            "message": "session started",
        }

    def stop_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id") or "default"
        with self.lock:
            existed = self.sessions.pop(session_id, None) is not None
        return {"ok": True, "session_id": session_id, "existed": existed}

    def tick(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id") or "default"
        project_case = payload.get("project_case") or "Auto_PUBG_ALL"
        game_case = payload.get("game_case") or "auto_pubg"
        screen = self._parse_screen(payload.get("screen"))

        with self.lock:
            self._ensure_loaded(project_case, game_case, screen)
            if session_id not in self.sessions:
                self.start_session(
                    {
                        "session_id": session_id,
                        "project_case": project_case,
                        "game_case": game_case,
                        "screen": payload.get("screen"),
                        "current_stage": payload.get("current_stage"),
                    }
                )
            session = self.sessions[session_id]
            if payload.get("current_stage") and payload.get("current_stage") != session.current_stage:
                session.current_stage = payload.get("current_stage")
            if screen:
                session.screen = screen

            frame = decode_base64_to_frame_rgb(payload["image"])
            stage_info = self.stage_controller.process_frame(frame, session.current_stage)
            worker = RemoteFrameWorker(session, frame, stage_info)
            self.logic_module.on_stage(worker)
            session.current_stage = worker.current_stage
            session.frame_index += 1

        return {
            "ok": True,
            "session_id": session_id,
            "frame_id": payload.get("frame_id"),
            "current_stage": session.current_stage,
            "stage_info": to_jsonable(stage_info),
            "actions": worker.actions,
            "stop": session.stopped,
        }

    def _ensure_loaded(self, project_case: str, game_case: str, screen: Optional[Tuple[int, int]]):
        key = (project_case, game_case)
        if self.loaded_key == key:
            if screen:
                os.environ["AUTOGAME_SCREEN_WIDTH"] = str(int(screen[0]))
                os.environ["AUTOGAME_SCREEN_HEIGHT"] = str(int(screen[1]))
                self._patch_resolution_helpers(screen)
            return

        os.environ["TARGET_PROJECT_CASE"] = project_case
        os.environ["TARGET_GAME_CASE"] = game_case
        if screen:
            os.environ["AUTOGAME_SCREEN_WIDTH"] = str(int(screen[0]))
            os.environ["AUTOGAME_SCREEN_HEIGHT"] = str(int(screen[1]))
            self._patch_resolution_helpers(screen)

        self.info_module = importlib.import_module(f"aw.autogame.customs_examples.{project_case}.info")
        game_scene_module = importlib.import_module("aw.autogame.tools.GameSceneHandler")
        logic_path = f"aw.autogame.customs_game_examples.{project_case}.{game_case}"
        self.logic_module = importlib.import_module(logic_path)
        self.stage_controller = game_scene_module.StageLogicController()
        self.loaded_key = key
        print(f"[RemoteBrain] loaded project={project_case}, game={game_case}")

    def _patch_resolution_helpers(self, screen: Tuple[int, int]):
        width, height = int(screen[0]), int(screen[1])

        def fake_get_resolution(r=True):
            return width, height

        def fake_get_wh():
            base_width = 768
            ratio = max(width, height) / max(1, min(width, height))
            return base_width, int(base_width * ratio)

        for module_name in (
            "aw.autogame.tools.Utils",
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.utils",
        ):
            try:
                module = importlib.import_module(module_name)
                module.get_resolution = fake_get_resolution
                if hasattr(module, "get_wh"):
                    module.get_wh = fake_get_wh
            except Exception as exc:
                print(f"[RemoteBrain] resolution patch skipped for {module_name}: {exc}")

    def _parse_screen(self, screen_payload) -> Optional[Tuple[int, int]]:
        if not screen_payload:
            return None
        width = screen_payload.get("width")
        height = screen_payload.get("height")
        if not width or not height:
            return None
        return int(width), int(height)

    def _first_active_stage(self, stage_dict: Dict[str, bool]) -> Optional[str]:
        for stage_name, active in stage_dict.items():
            if active:
                return stage_name
        return next(iter(stage_dict.keys()), None)


CORE = RemoteBrainCore()


class RemoteBrainHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"ok": True, "message": "remote brain alive"})
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path == "/session/start":
                response = CORE.start_session(payload)
            elif self.path == "/session/stop":
                response = CORE.stop_session(payload)
            elif self.path == "/tick":
                response = CORE.tick(payload)
            else:
                self._send_json({"ok": False, "error": "not found"}, status=404)
                return
            self._send_json(response)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json_loads(self.rfile.read(length))

    def _send_json(self, payload: Dict[str, Any], status: int = 200):
        data = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"[RemoteBrainHTTP] {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="auto_game remote brain server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RemoteBrainHandler)
    print(f"[RemoteBrain] serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
