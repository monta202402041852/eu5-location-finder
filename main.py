from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import keyboard
import mss
import numpy as np
import pytesseract
from PIL import Image, ImageTk
from rapidfuzz import fuzz
import tkinter as tk
import winsound

REGION_FILE = Path("region.json")
DEFAULT_TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
HOTKEY_MAP = {
    "F5": "toggle_fast_mode",
    "F6": "clear_alert",
    "F7": "set_search_term",
    "F8": "toggle_monitor",
    "ctrl+shift+m": "toggle_monitor",
    "F9": "record_left_top",
    "F10": "record_right_bottom_and_save",
}

NORMAL_INTERVAL = 0.18
FAST_INTERVAL = 0.06
FUZZ_THRESHOLD_DEFAULT = 86

DEFAULT_BINARY_OCR = False
DEFAULT_LINE_SPLIT_COUNT = 6


class Capturer:
    def grab(self, region: "Region") -> np.ndarray:
        raise NotImplementedError


class MSSCapturer(Capturer):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        logging.info("event=capturer type=mss status=lazy_initialized")

    def grab(self, region: "Region") -> np.ndarray:
        width = region.right - region.left
        height = region.bottom - region.top
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid_region left={region.left} top={region.top} right={region.right} bottom={region.bottom}")

        monitor = {
            "left": int(region.left),
            "top": int(region.top),
            "width": int(width),
            "height": int(height),
        }

        with self._lock:
            with mss.mss() as sct:
                raw = sct.grab(monitor)

        frame = np.asarray(raw)
        if frame.size == 0:
            raise RuntimeError("empty_frame size=0")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"unexpected_frame_shape shape={frame.shape}")

        return frame[:, :, :3].copy()


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def set_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        logging.exception("event=dpi_awareness status=failed")


def setup_logging() -> Path:
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"app_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging.info("event=logging_initialized log_file=%s", log_path)
    return log_path


def log_startup_environment() -> None:
    region_candidates = [Path.cwd() / REGION_FILE, Path(__file__).resolve().parent / REGION_FILE]
    region_status = ", ".join(
        f"{candidate} exists={candidate.exists()}" for candidate in region_candidates
    )
    tesseract_path = Path(DEFAULT_TESSERACT_PATH)
    logging.info("event=startup python_version=%s", platform.python_version())
    logging.info("event=startup os=%s", platform.platform())
    logging.info("event=startup cwd=%s", Path.cwd())
    logging.info("event=startup main_path=%s", Path(__file__).resolve())
    logging.info("event=startup region_candidates=%s", region_status)
    logging.info(
        "event=startup tesseract_configured_path=%s exists=%s",
        DEFAULT_TESSERACT_PATH,
        tesseract_path.exists(),
    )
    logging.info(
        "event=startup hotkeys=%s",
        ", ".join(f"{key}:{action}" for key, action in HOTKEY_MAP.items()),
    )


def get_cursor_pos() -> Tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


@dataclass
class Region:
    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def is_valid(self) -> bool:
        return self.right > self.left and self.bottom > self.top


class EU5LocationFinder:
    def __init__(self) -> None:
        self.state_lock = threading.RLock()
        self.region: Optional[Region] = self.load_region()
        self.corner_lt: Optional[Tuple[int, int]] = None
        self.corner_rb: Optional[Tuple[int, int]] = None
        self.search_term: str = ""
        self.monitoring = False
        self.fast_mode = False
        self.alert_active = False
        self.fuzz_threshold = FUZZ_THRESHOLD_DEFAULT
        self.recent_matches = deque(maxlen=4)
        self.last_ocr_text = ""
        self.last_capture_ok = False
        self.last_capture_reason = "not_started"
        self.capture_fail_streak = 0
        self.last_heartbeat_time = 0.0
        self.monitor_thread_started = False
        self.pending_pick_corner: Optional[str] = None
        self.pick_armed = False
        self.last_lbutton_down = False

        self.camera: Capturer = MSSCapturer()
        self.monitor_thread = threading.Thread(
            target=self.monitor_loop,
            daemon=True,
            name="monitor_loop_thread",
        )
        self.monitor_thread.start()
        self.monitor_thread_started = True
        logging.info("event=thread status=started name=%s", self.monitor_thread.name)

        self.root = tk.Tk()
        self.root.withdraw()
        self.overlay: Optional[tk.Toplevel] = None
        self.region_overlay: Optional[tk.Toplevel] = None
        self.region_overlay_canvas: Optional[tk.Canvas] = None
        self.region_overlay_visible = False
        self.overlay_image_label: Optional[tk.Label] = None
        self.overlay_text_label: Optional[tk.Label] = None
        self.term_status_window: Optional[tk.Toplevel] = None
        self.term_status_label: Optional[tk.Label] = None
        self.control_panel: Optional[tk.Toplevel] = None
        self.control_info_label: Optional[tk.Label] = None
        self.monitor_button: Optional[tk.Button] = None
        self.fast_button: Optional[tk.Button] = None
        self.term_dialog_window: Optional[tk.Toplevel] = None
        self.term_entry: Optional[tk.Entry] = None
        self.dialog_previous_active_hwnd: Optional[int] = None
        self.overlay_photo = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_image_label: Optional[tk.Label] = None
        self.preview_photo = None
        self.request_term_dialog = False
        self.request_clear_alert = False
        self.pending_alert: Optional[Tuple[np.ndarray, str, int]] = None
        self.resume_monitor_after_input = False
        self.ocr_binary_enabled = DEFAULT_BINARY_OCR
        self.ocr_line_split_count = DEFAULT_LINE_SPLIT_COUNT

        if os.path.exists(DEFAULT_TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH
            logging.info("event=tesseract status=configured path=%s", DEFAULT_TESSERACT_PATH)
        else:
            logging.warning("event=tesseract status=missing path=%s", DEFAULT_TESSERACT_PATH)

        self.setup_hotkeys()
        self.create_control_panel()
        self.show_region_overlay()
        self.root.after(50, self.ui_tick)
        self.print_status("Ready. F7=word F8=monitor F9/F10=region F5=fast F6=clear")

    def print_status(self, msg: str) -> None:
        logging.info("event=status message=%s", msg)

    def setup_hotkeys(self) -> None:
        keyboard.add_hotkey("F9", self.record_left_top)
        keyboard.add_hotkey("F10", self.record_right_bottom_and_save)
        keyboard.add_hotkey("F7", self.set_search_term)
        keyboard.add_hotkey("F8", self.toggle_monitor)
        keyboard.add_hotkey("ctrl+shift+m", self.toggle_monitor)
        keyboard.add_hotkey("F5", self.toggle_fast_mode)
        keyboard.add_hotkey("F6", self.clear_alert)
        logging.info("event=hotkeys_registered keys=%s", ",".join(HOTKEY_MAP.keys()))
        logging.info("hotkeys registered")

    def log_hotkey(self, key: str) -> None:
        logging.info("HOTKEY %s pressed", key)

    def load_region(self) -> Optional[Region]:
        if not REGION_FILE.exists():
            logging.warning("region.json not found. Please calibrate with F9/F10.")
            return None
        try:
            data = json.loads(REGION_FILE.read_text(encoding="utf-8"))
            region = Region(data["left"], data["top"], data["right"], data["bottom"])
            if not region.is_valid():
                raise ValueError("invalid rectangle")
            logging.info("Loaded region: %s", self.format_region(region))
            return region
        except Exception as e:
            logging.error("Failed to load region.json: %s", e)
            return None

    def save_region(self) -> None:
        if not self.region or not self.region.is_valid():
            self.print_status("Cannot save invalid region.")
            return
        REGION_FILE.write_text(
            json.dumps(self.region.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.print_status(f"Saved region: {self.format_region(self.region)}")
        self.log_region_coordinates("saved")

    def format_region(self, region: Optional[Region]) -> str:
        if region is None:
            return "(unset)"
        return f"left={region.left}, top={region.top}, right={region.right}, bottom={region.bottom}"

    def log_region_coordinates(self, source: str) -> None:
        with self.state_lock:
            region = self.region
        logging.info("event=region source=%s coords=%s", source, self.format_region(region))

    def record_left_top(self) -> None:
        self.log_hotkey("F9")
        try:
            self.corner_lt = get_cursor_pos()
            self.print_status(f"F9 captured left-top: {self.corner_lt}")
            self.log_region_coordinates("f9")
        except Exception:
            logging.exception("event=hotkey_error key=F9")

    def record_right_bottom_and_save(self) -> None:
        self.log_hotkey("F10")
        try:
            self.corner_rb = get_cursor_pos()
            self.print_status(f"F10 captured right-bottom: {self.corner_rb}")
            self.log_region_coordinates("f10")
            if not self.corner_lt:
                self.print_status("Left-top is missing. Press F9 first.")
                return
            left = min(self.corner_lt[0], self.corner_rb[0])
            top = min(self.corner_lt[1], self.corner_rb[1])
            right = max(self.corner_lt[0], self.corner_rb[0])
            bottom = max(self.corner_lt[1], self.corner_rb[1])
            region = Region(left, top, right, bottom)
            if not region.is_valid():
                self.print_status("Invalid region; capture again.")
                return
            self.region = region
            self.save_region()
            self.show_region_overlay()
        except Exception:
            logging.exception("event=hotkey_error key=F10")

    def set_search_term(self) -> None:
        self.log_hotkey("F7")
        try:
            with self.state_lock:
                self.request_term_dialog = True
            logging.info("event=search_term_dialog status=requested")
        except Exception:
            logging.exception("event=hotkey_error key=F7")

    def toggle_fast_mode(self, source: str = "hotkey") -> None:
        if source == "hotkey":
            self.log_hotkey("F5")
        try:
            with self.state_lock:
                self.fast_mode = not self.fast_mode
                fast_mode = self.fast_mode
            self.print_status(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
            logging.info("fast=%s source=%s", "ON" if fast_mode else "OFF", source)
        except Exception:
            logging.exception("event=hotkey_error key=F5")

    def toggle_monitor(self, source: str = "hotkey") -> None:
        if source == "hotkey":
            self.log_hotkey("F8")
        try:
            with self.state_lock:
                alert_active = self.alert_active
                region = self.region
                search_term = self.search_term
            if alert_active:
                self.print_status("Alert active. Press F6 to clear first.")
                return
            if not region:
                self.print_status("Region not configured. Use F9/F10 first.")
                return
            if not search_term:
                self.print_status("Search term is empty. Use F7 first.")
                return
            self.set_monitoring(not self.monitoring, source=source)
        except Exception:
            logging.exception("event=hotkey_error key=F8")

    def set_monitoring(self, value: bool, source: str) -> None:
        with self.state_lock:
            self.monitoring = value
            monitoring = self.monitoring
            self.recent_matches.clear()
            self.capture_fail_streak = 0
        self.print_status(f"Monitoring: {'ON' if monitoring else 'OFF'}")
        logging.info("monitor=%s source=%s", "ON" if monitoring else "OFF", source)

    def clear_alert(self) -> None:
        self.log_hotkey("F6")
        try:
            with self.state_lock:
                self.request_clear_alert = True
            logging.info("event=alert_clear status=requested")
        except Exception:
            logging.exception("event=hotkey_error key=F6")

    def monitor_loop(self) -> None:
        logging.info("event=thread_loop status=running name=monitor_loop_thread")
        try:
            while True:
                try:
                    with self.state_lock:
                        monitoring = self.monitoring
                        alert_active = self.alert_active
                        fast_mode = self.fast_mode
                        region = self.region
                        term = self.search_term
                    if monitoring and not alert_active:
                        frame, cap_ok, cap_reason = self.capture_region()
                        with self.state_lock:
                            self.last_capture_ok = cap_ok
                            self.last_capture_reason = cap_reason or "ok"
                        if frame is not None:
                            hit, score, token = self.detect_term(frame)
                            with self.state_lock:
                                self.recent_matches.append((hit, score, token, frame))
                                should_fire = self.should_fire_alert()
                                best_candidate = (
                                    max(self.recent_matches, key=lambda x: x[1])
                                    if should_fire
                                    else None
                                )
                            if best_candidate is not None:
                                _, best_score, best_token, best_frame = best_candidate
                                self.on_hit(best_frame, best_token, best_score)
                        self.emit_heartbeat(
                            monitoring=monitoring,
                            fast_mode=fast_mode,
                            region_set=region is not None,
                            term=term,
                            capture_ok=cap_ok,
                            capture_reason=cap_reason,
                            frame=frame,
                        )
                        interval = FAST_INTERVAL if fast_mode else NORMAL_INTERVAL
                        time.sleep(0.05 if not cap_ok else interval)
                    else:
                        time.sleep(0.05)
                except Exception:
                    logging.exception("event=monitor_loop_error")
                    time.sleep(0.2)
        except Exception:
            logging.exception("event=thread status=stopped name=monitor_loop_thread reason=exception")

    def emit_heartbeat(
        self,
        monitoring: bool,
        fast_mode: bool,
        region_set: bool,
        term: str,
        capture_ok: bool,
        capture_reason: str,
        frame: Optional[np.ndarray],
    ) -> None:
        now = time.time()
        if now - self.last_heartbeat_time < 1.0:
            return
        self.last_heartbeat_time = now
        logging.info(
            'HEARTBEAT monitor=%s fast=%s region=%s coords="%s" term="%s"',
            "ON" if monitoring else "OFF",
            "ON" if fast_mode else "OFF",
            "SET" if region_set else "UNSET",
            self.format_region(self.region),
            term,
        )
        if capture_ok and frame is not None:
            h, w = frame.shape[:2]
            logging.info("CAPTURE ok=True size=%sx%s", w, h)
        else:
            logging.info('CAPTURE ok=False reason="%s"', capture_reason)
        sample = self.last_ocr_text[:30].replace("\n", " ")
        logging.info('OCR chars=%s sample="%s"', len(self.last_ocr_text), sample)

    def should_fire_alert(self) -> bool:
        if not self.recent_matches:
            return False
        if any(hit for hit, _, _, _ in self.recent_matches):
            return True
        high_scores = [s for _, s, _, _ in self.recent_matches if s >= self.fuzz_threshold + 7]
        return len(high_scores) >= 2

    def capture_region(self) -> Tuple[Optional[np.ndarray], bool, str]:
        if not self.region:
            return None, False, "region_unset"

        reason = "unknown"
        for attempt in range(3):
            try:
                frame = self.camera.grab(region=self.region)
                if frame.size == 0:
                    reason = "size_zero"
                elif frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    reason = f"invalid_size:{frame.shape[1]}x{frame.shape[0]}"
                elif np.mean(frame) < 1.0:
                    reason = "black_frame"
                else:
                    width = int(frame.shape[1])
                    height = int(frame.shape[0])
                    with self.state_lock:
                        self.capture_fail_streak = 0
                    return frame, True, f"size={width}x{height}"

                logging.warning("event=capture status=retry attempt=%s reason=%s", attempt + 1, reason)
            except Exception as exc:
                reason = f"exception:{exc}"
                logging.exception("event=capture status=retry attempt=%s reason=%s", attempt + 1, reason)

            time.sleep(0.05)

        with self.state_lock:
            self.capture_fail_streak += 1
            streak = self.capture_fail_streak
        if streak < 3:
            logging.warning("event=capture status=soft_fail streak=%s reason=%s", streak, reason)
            return None, True, f"retrying:{reason}"

        logging.error("event=capture status=failed streak=%s reason=%s", streak, reason)
        return None, False, f"CAPTURE FAIL:{reason}"

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enlarged, 1.5, blurred, -0.5, 0)

        if self.ocr_binary_enabled:
            _, sharpened = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return sharpened

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("•", "・").replace("･", "・")
        text = re.sub(r"[\s\u3000]+", "", text)
        return text

    def _ocr_with_psm(self, image: np.ndarray, psm: int) -> str:
        config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=0"
        return pytesseract.image_to_string(image, lang="jpn", config=config)

    def _split_lines(self, image: np.ndarray) -> list[np.ndarray]:
        lines: list[np.ndarray] = []
        h, _ = image.shape[:2]
        line_count = max(1, self.ocr_line_split_count)
        step = max(1, h // line_count)
        for i in range(line_count):
            top = i * step
            bottom = h if i == line_count - 1 else min(h, (i + 1) * step)
            if bottom - top >= 4:
                lines.append(image[top:bottom, :])
        return lines

    def ocr_tokens(self, image: np.ndarray) -> list[str]:
        try:
            raw_psm7 = self._ocr_with_psm(image, psm=7)
            raw_psm6 = self._ocr_with_psm(image, psm=6)
            picked = raw_psm7 if len(raw_psm7.strip()) >= len(raw_psm6.strip()) else raw_psm6

            line_tokens: list[str] = []
            for line_img in self._split_lines(image):
                line_raw = self._ocr_with_psm(line_img, psm=7)
                line_tokens.extend(self.normalize_text(p) for p in re.split(r"[\r\n]+", line_raw) if p.strip())

            tokens = [self.normalize_text(p) for p in re.split(r"[\r\n]+", picked)]
            tokens = [p for p in tokens if p]
            tokens.extend([t for t in line_tokens if t])
            deduped = list(dict.fromkeys(tokens))
            return deduped
        except Exception:
            logging.exception("OCR error")
            raise

    def detect_term(self, frame: np.ndarray) -> Tuple[bool, int, str]:
        try:
            prep = self.preprocess(frame)
            tokens = self.ocr_tokens(prep)
            self.last_ocr_text = " ".join(tokens)
            if not tokens:
                return False, 0, ""

            with self.state_lock:
                search_term = self.search_term
                fuzz_threshold = self.fuzz_threshold

            for token in tokens:
                if token == search_term:
                    return True, 100, token

            best_score = 0
            best_token = ""
            for token in tokens:
                score = fuzz.partial_ratio(search_term, token)
                if score > best_score:
                    best_score = score
                    best_token = token

            return best_score >= fuzz_threshold, int(best_score), best_token
        except Exception:
            logging.exception("event=detect_term status=failed")
            return False, 0, ""

    def on_hit(self, frame: np.ndarray, token: str, score: int) -> None:
        with self.state_lock:
            self.monitoring = False
            self.alert_active = True
            self.pending_alert = (frame, token, score)
        self.print_status(f"HIT: token={token} score={score}")
        try:
            winsound.Beep(1800, 300)
            winsound.Beep(2200, 500)
        except Exception:
            logging.exception("event=beep status=failed")

    def create_term_status_window(self) -> None:
        # Kept for backward compatibility.
        self.create_control_panel()

    def create_control_panel(self) -> None:
        self.control_panel = tk.Toplevel(self.root)
        panel = self.control_panel
        panel.title("EU5 Control")
        panel.attributes("-topmost", True)
        panel.resizable(False, False)
        panel.configure(bg="#1e1e1e")
        panel.geometry("430x320+20+20")

        self.control_info_label = tk.Label(
            panel,
            text="",
            font=("Meiryo", 9),
            fg="#ffffff",
            bg="#1e1e1e",
            justify="left",
            anchor="w",
        )
        self.control_info_label.pack(fill="x", padx=8, pady=(8, 4))

        row1 = tk.Frame(panel, bg="#1e1e1e")
        row1.pack(fill="x", padx=8, pady=2)
        tk.Button(row1, text="Set Term", width=11, command=self.set_search_term).pack(side="left", padx=2)
        tk.Button(row1, text="Pick LT", width=11, command=self.pick_left_top).pack(side="left", padx=2)
        tk.Button(row1, text="Pick RB", width=11, command=self.pick_right_bottom).pack(side="left", padx=2)

        row2 = tk.Frame(panel, bg="#1e1e1e")
        row2.pack(fill="x", padx=8, pady=2)
        self.monitor_button = tk.Button(row2, text="Monitor ON/OFF", width=17, command=self.toggle_monitor_ui)
        self.monitor_button.pack(side="left", padx=2)
        self.fast_button = tk.Button(row2, text="Fast ON/OFF", width=17, command=self.toggle_fast_mode_ui)
        self.fast_button.pack(side="left", padx=2)

        row3 = tk.Frame(panel, bg="#1e1e1e")
        row3.pack(fill="x", padx=8, pady=2)
        tk.Button(row3, text="Show Region", width=17, command=self.show_region_overlay).pack(side="left", padx=2)
        tk.Button(row3, text="Hide Region", width=17, command=self.hide_region_overlay).pack(side="left", padx=2)

        row4 = tk.Frame(panel, bg="#1e1e1e")
        row4.pack(fill="x", padx=8, pady=2)
        tk.Button(row4, text="Clear Alert", width=17, command=self.clear_alert).pack(side="left", padx=2)
        tk.Button(row4, text="Test Alert", width=17, command=self.test_alert).pack(side="left", padx=2)

        row5 = tk.Frame(panel, bg="#1e1e1e")
        row5.pack(fill="x", padx=8, pady=2)
        tk.Button(row5, text="Preview", width=17, command=self.preview_capture).pack(side="left", padx=2)
        tk.Button(row5, text="Quit", width=17, command=self.root.destroy).pack(side="left", padx=2)

    def preview_capture(self) -> None:
        frame, cap_ok, cap_reason = self.capture_region()
        with self.state_lock:
            self.last_capture_ok = cap_ok
            self.last_capture_reason = cap_reason
        if frame is None:
            self.print_status(f"Preview failed: {cap_reason}")
            return
        self.show_preview_window(frame)

    def show_preview_window(self, frame: np.ndarray) -> None:
        if self.preview_window and not self.preview_window.winfo_exists():
            self.preview_window = None
            self.preview_image_label = None

        if self.preview_window is None:
            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.title("Capture Preview")
            self.preview_window.attributes("-topmost", True)
            self.preview_window.configure(bg="#111111")
            self.preview_image_label = tk.Label(self.preview_window, bg="#111111")
            self.preview_image_label.pack(padx=8, pady=8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.preview_photo = ImageTk.PhotoImage(pil)

        if self.preview_image_label:
            self.preview_image_label.configure(image=self.preview_photo)

        self.preview_window.deiconify()
        self.preview_window.lift()
        self.print_status(f"Preview updated: size={frame.shape[1]}x{frame.shape[0]}")

    def test_alert(self) -> None:
        frame = np.zeros((90, 360, 3), dtype=np.uint8)
        cv2.putText(frame, "TEST ALERT", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        with self.state_lock:
            self.alert_active = True
            self.pending_alert = (frame, "TEST_ALERT", 100)
        try:
            winsound.Beep(1500, 200)
            winsound.Beep(2200, 200)
            winsound.Beep(2600, 350)
        except Exception:
            logging.exception("event=beep status=failed")
        self.print_status("Test Alert fired.")

    def hide_region_overlay(self) -> None:
        self.region_overlay_visible = False
        if self.region_overlay:
            self.region_overlay.withdraw()
        self.print_status("Region overlay hidden.")

    def show_region_overlay(self) -> None:
        self.region_overlay_visible = True
        self.update_region_overlay()

    def update_region_overlay(self) -> None:
        if not self.region_overlay_visible:
            return
        region = self.region
        if region is None or not region.is_valid():
            if self.region_overlay:
                self.region_overlay.withdraw()
            return

        width = max(1, region.right - region.left)
        height = max(1, region.bottom - region.top)

        if self.region_overlay is None:
            overlay = tk.Toplevel(self.root)
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.attributes("-alpha", 0.25)
            overlay.configure(bg="black")

            canvas = tk.Canvas(overlay, bg="black", highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)

            self.region_overlay = overlay
            self.region_overlay_canvas = canvas

        self.region_overlay.geometry(f"{width}x{height}+{region.left}+{region.top}")
        self.region_overlay.deiconify()
        self.region_overlay.lift()

        if self.region_overlay_canvas:
            self.region_overlay_canvas.configure(width=width, height=height)
            self.region_overlay_canvas.delete("all")
            self.region_overlay_canvas.create_rectangle(
                1,
                1,
                width - 2,
                height - 2,
                outline="#ff0000",
                width=3,
            )
            self.region_overlay_canvas.create_text(
                8,
                8,
                text=f"region({region.left}, {region.top}, {region.right}, {region.bottom})",
                fill="#ff0000",
                anchor="nw",
                font=("Meiryo", 10, "bold"),
            )

    def toggle_monitor_ui(self) -> None:
        self.toggle_monitor(source="ui")

    def toggle_fast_mode_ui(self) -> None:
        self.toggle_fast_mode(source="ui")

    def pick_left_top(self) -> None:
        self.pending_pick_corner = "lt"
        self.pick_armed = False
        self.print_status("Pick LT armed: click target left-top point.")

    def pick_right_bottom(self) -> None:
        self.pending_pick_corner = "rb"
        self.pick_armed = False
        self.print_status("Pick RB armed: click target right-bottom point.")

    def poll_point_pick(self) -> None:
        down = bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
        if self.pending_pick_corner and not self.pick_armed and not down:
            self.pick_armed = True

        if self.pending_pick_corner and self.pick_armed and down and not self.last_lbutton_down:
            x, y = get_cursor_pos()
            if self.pending_pick_corner == "lt":
                self.corner_lt = (x, y)
                self.print_status(f"LT captured: {self.corner_lt}")
            elif self.pending_pick_corner == "rb":
                self.corner_rb = (x, y)
                self.print_status(f"RB captured: {self.corner_rb}")
                if not self.corner_lt:
                    self.print_status("Left-top is missing. Capture LT first.")
                else:
                    left = min(self.corner_lt[0], self.corner_rb[0])
                    top = min(self.corner_lt[1], self.corner_rb[1])
                    right = max(self.corner_lt[0], self.corner_rb[0])
                    bottom = max(self.corner_lt[1], self.corner_rb[1])
                    region = Region(left, top, right, bottom)
                    if region.is_valid():
                        self.region = region
                        self.save_region()
                        self.show_region_overlay()
                    else:
                        self.print_status("Invalid region; capture LT/RB again.")
            self.pending_pick_corner = None
            self.pick_armed = False

        self.last_lbutton_down = down

    def refresh_control_panel(self) -> None:
        if not self.control_info_label:
            return
        with self.state_lock:
            term = self.search_term or "(unset)"
            region = self.region
            monitor_txt = "ON" if self.monitoring else "OFF"
            fast_txt = "ON" if self.fast_mode else "OFF"
            cap_txt = "ok" if self.last_capture_ok else "FAIL"
            cap_reason = self.last_capture_reason
            ocr_chars = len(self.last_ocr_text)
            sample = self.last_ocr_text[:36].replace("\n", " ")
        pick_state = self.pending_pick_corner.upper() if self.pending_pick_corner else "-"
        region_txt = self.format_region(region)
        self.control_info_label.config(
            text=(
                f"TERM: {term}\n"
                f"REGION: {region_txt}\n"
                f"MONITOR: {monitor_txt}  FAST: {fast_txt}\n"
                f"CAPTURE: {cap_txt} ({cap_reason})\n"
                f"OCR chars: {ocr_chars}\n"
                f"LAST OCR sample: {sample}\n"
                f"PICK: {pick_state}"
            ),
            fg="#ff6060" if cap_txt == "FAIL" else "#ffffff",
        )

    def ui_tick(self) -> None:
        self.poll_point_pick()
        self.process_ui_requests()
        self.refresh_control_panel()
        self.root.after(50, self.ui_tick)

    def process_ui_requests(self) -> None:
        with self.state_lock:
            request_term_dialog = self.request_term_dialog
            if request_term_dialog:
                self.request_term_dialog = False
                self.resume_monitor_after_input = self.monitoring
                self.monitoring = False

            request_clear_alert = self.request_clear_alert
            if request_clear_alert:
                self.request_clear_alert = False

            pending_alert = self.pending_alert
            if pending_alert is not None:
                self.pending_alert = None

            search_term = self.search_term

        if self.term_status_label:
            status_text = f"現在の検索語: {search_term}" if search_term else "現在の検索語: (未設定)"
            self.term_status_label.config(text=status_text)

        if request_term_dialog:
            self.show_term_dialog()

        if request_clear_alert:
            if self.overlay:
                self.overlay.destroy()
                self.overlay = None
            with self.state_lock:
                self.alert_active = False
            self.print_status("Alert cleared.")

        if pending_alert is not None:
            frame, token, score = pending_alert
            self.show_overlay(frame, token, score)

        self.update_region_overlay()

    def show_term_dialog(self) -> None:
        if self.term_dialog_window is not None:
            self.term_dialog_window.lift()
            if self.term_entry:
                self.term_entry.focus_set()
            return

        try:
            self.dialog_previous_active_hwnd = int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            self.dialog_previous_active_hwnd = None

        win = tk.Toplevel(self.root)
        self.term_dialog_window = win
        win.title("検索語入力")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        win.configure(bg="#1f1f1f")

        tk.Label(
            win,
            text="検索語（1語）を入力",
            font=("Meiryo", 11, "bold"),
            fg="#ffffff",
            bg="#1f1f1f",
        ).pack(padx=12, pady=(10, 6), anchor="w")

        entry = tk.Entry(win, width=28, font=("Meiryo", 12))
        with self.state_lock:
            entry.insert(0, self.search_term)
        entry.pack(padx=12, pady=(0, 10))
        self.term_entry = entry

        def submit() -> None:
            term = self.normalize_text(entry.get())
            with self.state_lock:
                self.search_term = term
                resume = self.resume_monitor_after_input and bool(self.search_term)
                self.monitoring = resume
                self.resume_monitor_after_input = False
            self.print_status(f"Search term set: {term}" if term else "Search term cleared.")
            self.close_term_dialog("ok")

        def cancel() -> None:
            with self.state_lock:
                resume = self.resume_monitor_after_input and bool(self.search_term)
                self.monitoring = resume
                self.resume_monitor_after_input = False
            self.close_term_dialog("cancel")

        entry.bind("<Return>", lambda _event: submit())
        win.bind("<Escape>", lambda _event: cancel())
        win.protocol("WM_DELETE_WINDOW", cancel)
        entry.focus_set()

    def close_term_dialog(self, reason: str) -> None:
        if self.term_dialog_window:
            try:
                self.term_dialog_window.grab_release()
            except Exception:
                pass
            self.term_dialog_window.destroy()
            self.term_dialog_window = None
            self.term_entry = None

        try:
            self.root.focus_force()
        except Exception:
            pass

        if self.dialog_previous_active_hwnd:
            try:
                ctypes.windll.user32.SetForegroundWindow(self.dialog_previous_active_hwnd)
            except Exception:
                pass
            self.dialog_previous_active_hwnd = None

        self.root.update_idletasks()
        logging.info("event=term_dialog status=closed reason=%s", reason)
        logging.info("event=hotkey_system status=alive")
        self.root.after(200, lambda: logging.info("post-term-dialog tick"))

    def show_overlay(self, frame: np.ndarray, token: str, score: int) -> None:
        if self.overlay:
            self.overlay.destroy()

        self.overlay = tk.Toplevel(self.root)
        self.overlay.title("EU5 Location Finder ALERT")
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg="#111111")

        self.overlay_text_label = tk.Label(
            self.overlay,
            text=f"検出語: {token}  (score={score})\n検索語: {self.search_term}\nF6で解除",
            font=("Meiryo", 14, "bold"),
            fg="#ff5050",
            bg="#111111",
            justify="left",
        )
        self.overlay_text_label.pack(padx=12, pady=12, anchor="w")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        scale = 2
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)
        self.overlay_photo = ImageTk.PhotoImage(pil)

        self.overlay_image_label = tk.Label(self.overlay, image=self.overlay_photo, bg="#111111")
        self.overlay_image_label.pack(padx=12, pady=(0, 12))

        self.overlay.protocol("WM_DELETE_WINDOW", lambda: None)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    setup_logging()
    log_startup_environment()
    set_dpi_aware()
    app = EU5LocationFinder()
    app.run()
