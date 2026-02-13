import ctypes
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import dxcam
import keyboard
import numpy as np
import pytesseract
from PIL import Image, ImageTk
from rapidfuzz import fuzz
import tkinter as tk
from tkinter import simpledialog, messagebox
import winsound

REGION_FILE = Path("region.json")
DEFAULT_TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

NORMAL_INTERVAL = 0.18
FAST_INTERVAL = 0.06
FUZZ_THRESHOLD_DEFAULT = 86

KATAKANA_WHITELIST = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンァィゥェォャュョッー・ヴガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ0123456789"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def set_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        logging.warning("Failed to enable DPI awareness.")


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
        self.region: Optional[Region] = self.load_region()
        self.corner_lt: Optional[Tuple[int, int]] = None
        self.corner_rb: Optional[Tuple[int, int]] = None
        self.search_term: str = ""
        self.monitoring = False
        self.fast_mode = False
        self.alert_active = False
        self.fuzz_threshold = FUZZ_THRESHOLD_DEFAULT
        self.recent_matches = deque(maxlen=4)

        self.camera = dxcam.create(output_color="BGR")
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.root = tk.Tk()
        self.root.withdraw()
        self.overlay: Optional[tk.Toplevel] = None
        self.overlay_image_label: Optional[tk.Label] = None
        self.overlay_text_label: Optional[tk.Label] = None
        self.overlay_photo = None

        if os.path.exists(DEFAULT_TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

        self.setup_hotkeys()
        self.print_status("Ready. F7=word F8=monitor F9/F10=region F5=fast F6=clear")

    def print_status(self, msg: str) -> None:
        logging.info(msg)
        print(msg)

    def setup_hotkeys(self) -> None:
        keyboard.add_hotkey("F9", self.record_left_top)
        keyboard.add_hotkey("F10", self.record_right_bottom_and_save)
        keyboard.add_hotkey("F7", self.set_search_term)
        keyboard.add_hotkey("F8", self.toggle_monitor)
        keyboard.add_hotkey("F5", self.toggle_fast_mode)
        keyboard.add_hotkey("F6", self.clear_alert)

    def load_region(self) -> Optional[Region]:
        if not REGION_FILE.exists():
            logging.warning("region.json not found. Please calibrate with F9/F10.")
            return None
        try:
            data = json.loads(REGION_FILE.read_text(encoding="utf-8"))
            region = Region(data["left"], data["top"], data["right"], data["bottom"])
            if not region.is_valid():
                raise ValueError("invalid rectangle")
            logging.info("Loaded region: %s", region)
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
        self.print_status(f"Saved region: {self.region}")

    def record_left_top(self) -> None:
        self.corner_lt = get_cursor_pos()
        self.print_status(f"F9 captured left-top: {self.corner_lt}")

    def record_right_bottom_and_save(self) -> None:
        self.corner_rb = get_cursor_pos()
        self.print_status(f"F10 captured right-bottom: {self.corner_rb}")
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

    def set_search_term(self) -> None:
        term = simpledialog.askstring("Search term", "検索語（1語）を入力してください:")
        if term is None:
            return
        self.search_term = self.normalize_text(term)
        self.print_status(f"Search term set: {self.search_term}")

    def toggle_fast_mode(self) -> None:
        self.fast_mode = not self.fast_mode
        self.print_status(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")

    def toggle_monitor(self) -> None:
        if self.alert_active:
            self.print_status("Alert active. Press F6 to clear first.")
            return
        if not self.region:
            self.print_status("Region not configured. Use F9/F10 first.")
            return
        if not self.search_term:
            self.print_status("Search term is empty. Use F7 first.")
            return
        self.monitoring = not self.monitoring
        self.recent_matches.clear()
        self.print_status(f"Monitoring: {'ON' if self.monitoring else 'OFF'}")

    def clear_alert(self) -> None:
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.alert_active = False
        self.print_status("Alert cleared.")

    def monitor_loop(self) -> None:
        while True:
            try:
                if self.monitoring and not self.alert_active:
                    frame = self.capture_region()
                    if frame is not None:
                        hit, score, token = self.detect_term(frame)
                        self.recent_matches.append((hit, score, token, frame))
                        if self.should_fire_alert():
                            _, best_score, best_token, best_frame = max(
                                self.recent_matches,
                                key=lambda x: x[1],
                            )
                            self.on_hit(best_frame, best_token, best_score)
                    interval = FAST_INTERVAL if self.fast_mode else NORMAL_INTERVAL
                    time.sleep(interval)
                else:
                    time.sleep(0.05)
            except Exception as e:
                logging.exception("Monitor loop error: %s", e)
                time.sleep(0.2)

    def should_fire_alert(self) -> bool:
        if not self.recent_matches:
            return False
        if any(hit for hit, _, _, _ in self.recent_matches):
            return True
        high_scores = [s for _, s, _, _ in self.recent_matches if s >= self.fuzz_threshold + 7]
        return len(high_scores) >= 2

    def capture_region(self) -> Optional[np.ndarray]:
        if not self.region:
            return None
        try:
            frame = self.camera.grab(region=self.region.as_tuple())
            if frame is None:
                logging.warning("Capture failed: empty frame.")
                return None
            if np.mean(frame) < 1.0:
                logging.warning("Capture appears black. Check borderless/window mode or admin rights.")
            return frame
        except Exception as e:
            logging.error("Capture failure: %s", e)
            return None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
        kernel = np.ones((2, 2), np.uint8)
        denoise = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        return denoise

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("•", "・").replace("･", "・")
        text = re.sub(r"[\s\u3000]+", "", text)
        return text

    def ocr_tokens(self, image: np.ndarray) -> list[str]:
        config = (
            "--oem 3 --psm 6 "
            f"-c tessedit_char_whitelist={KATAKANA_WHITELIST} "
            "-c preserve_interword_spaces=0"
        )
        raw = pytesseract.image_to_string(image, lang="jpn", config=config)
        pieces = [self.normalize_text(p) for p in re.split(r"[\r\n]+", raw)]
        return [p for p in pieces if p]

    def detect_term(self, frame: np.ndarray) -> Tuple[bool, int, str]:
        try:
            prep = self.preprocess(frame)
            tokens = self.ocr_tokens(prep)
            if not tokens:
                return False, 0, ""

            for token in tokens:
                if token == self.search_term:
                    return True, 100, token

            best_score = 0
            best_token = ""
            for token in tokens:
                score = fuzz.partial_ratio(self.search_term, token)
                if score > best_score:
                    best_score = score
                    best_token = token

            return best_score >= self.fuzz_threshold, int(best_score), best_token
        except Exception as e:
            logging.error("OCR error: %s", e)
            return False, 0, ""

    def on_hit(self, frame: np.ndarray, token: str, score: int) -> None:
        self.monitoring = False
        self.alert_active = True
        self.print_status(f"HIT: token={token} score={score}")
        try:
            winsound.Beep(1800, 300)
            winsound.Beep(2200, 500)
        except Exception:
            pass
        self.root.after(0, lambda: self.show_overlay(frame, token, score))

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    set_dpi_aware()
    app = EU5LocationFinder()
    app.run()
