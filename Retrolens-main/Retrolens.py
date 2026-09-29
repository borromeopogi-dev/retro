from dataclasses import dataclass
import random
import time
import threading
from typing import Dict, List, Tuple, Callable

# Added image asset (kept separate so the existing code/behavior is unchanged).
RAIN_IMAGE_PATH = "Screenshot 2026-08-28 111828.png"
import os
import shutil
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np

# Python audio playback for MOV/AAC audio using PyAV + sounddevice.
try:
    import av
    import sounddevice as sd
    PYTHON_AUDIO_AVAILABLE = True
except ImportError:
    av = None
    sd = None
    PYTHON_AUDIO_AVAILABLE = False


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    pinch_threshold_px: float = 45.0
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_dist_threshold_px: float = 80.0
    portal_video_path: str = "IMG_7343(1).MOV"


class FilterBank:
    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)

    @staticmethod
    def video_look(roi: np.ndarray, ghost: np.ndarray | None = None) -> np.ndarray:
        """Purple cinematic / soft double-exposure look inspired by the supplied video."""
        base = roi.astype(np.float32)

        # Purple/magenta lighting similar to the reference clip.
        b, g, r = cv2.split(base)
        b = b * 1.08
        g = g * 0.62
        r = r * 1.12
        purple = cv2.merge([b, g, r])
        purple = np.clip(purple, 0, 255).astype(np.uint8)

        # Slight bloom keeps highlights soft instead of overly sharp.
        glow = cv2.GaussianBlur(purple, (0, 0), 9)
        purple = cv2.addWeighted(purple, 0.82, glow, 0.18, 0)

        # Temporal ghost/double exposure.
        if ghost is not None and ghost.shape == purple.shape:
            ghost = cv2.GaussianBlur(ghost, (0, 0), 2)
            purple = cv2.addWeighted(purple, 0.78, ghost, 0.22, 0)

        return purple


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_fist_closed(landmarks, w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
        tips = [8, 12, 16, 20]
        distances = [np.linalg.norm(np.array([landmarks[t].x * w, landmarks[t].y * h]) - wrist) for t in tips]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx, dy = index[0] - thumb[0], index[1] - thumb[1]
        return (dy > 25) or (abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[1], rightmost[0], leftmost[1]], dtype=np.int32)



class VideoPortal:
    """Smooth, non-blocking looping video reader for the portal."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(self.fps) or self.fps <= 1:
            self.fps = 30.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.start_time = time.perf_counter()
        self.last_index = -1
        self.last_frame: np.ndarray | None = None

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open portal video: {path}")

        print(
            f"[VIDEO] Opened: {os.path.basename(path)} | "
            f"{self.fps:.1f} FPS | {self.total_frames} frames"
        )

    def _restart(self) -> bool:
        """Safely rewind the video once."""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.last_index = -1
            return True
        except Exception:
            return False

    def get_frame(self) -> np.ndarray | None:
        """
        Return the video frame for the current time.

        IMPORTANT: this never uses an unbounded while-loop. If the MOV decoder
        fails, it returns the last good frame instead of freezing the program.
        """
        if not self.cap.isOpened():
            return self.last_frame

        if self.total_frames > 0:
            elapsed = time.perf_counter() - self.start_time
            wanted = int(elapsed * self.fps) % self.total_frames

            # Same video frame as last call: reuse it instead of returning None.
            if wanted == self.last_index and self.last_frame is not None:
                return self.last_frame

            # Normal forward playback.
            if wanted > self.last_index:
                ok = True
                frame = None

                # Read only the required number of frames. This is bounded.
                steps = min(wanted - self.last_index, 8)
                for _ in range(steps):
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        break
                    self.last_index += 1

                # If the requested frame is far ahead, seek directly rather
                # than blocking while trying to decode hundreds of frames.
                if wanted - self.last_index > 0:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
                        ok, frame = self.cap.read()
                        if ok and frame is not None:
                            self.last_index = wanted
                    except Exception:
                        ok = False

                if ok and frame is not None:
                    self.last_frame = frame
                    return frame

            # Loop or recover from a decoder failure.
            self._restart()
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.last_index = 0
                self.last_frame = frame
                return frame

            return self.last_frame

        # Unknown frame count: just read one frame and recover once on EOF.
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.last_frame = frame
            return frame

        self._restart()
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.last_frame = frame
            return frame

        return self.last_frame

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


class PortalAudio:
    """
    Reliable continuous playback of the MOV's embedded audio.

    PyAV decodes the audio stream and sounddevice OutputStream sends one
    continuous PCM stream to the Windows default output device.  Using
    sd.play() once per decoded frame can create gaps/clicks or appear silent,
    so this class keeps one OutputStream open for the whole VIDEO filter.
    """

    def __init__(self, path: str):
        self.path = path
        self.thread = None
        self.stop_event = threading.Event()
        self.running = False
        self.stream = None
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return

        if not PYTHON_AUDIO_AVAILABLE:
            print("[AUDIO] Missing Python audio packages.")
            print("[AUDIO] Install once with: python -m pip install av sounddevice")
            return

        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(
            target=self._play_loop,
            name="PortalAudio",
            daemon=True,
        )
        self.thread.start()

    def _open_output(self):
        """Open the Windows default output as one continuous stream."""
        device = sd.default.device
        output_device = None

        if isinstance(device, (list, tuple)) and len(device) >= 2:
            output_device = device[1]
        elif isinstance(device, int):
            output_device = device

        print(f"[AUDIO] Output device: {output_device if output_device is not None else 'default'}")

        return sd.OutputStream(
            samplerate=48000,
            channels=2,
            dtype="float32",
            device=output_device,
            blocksize=0,
        )

    def _play_loop(self):
        container = None
        stream = None
        try:
            while not self.stop_event.is_set():
                try:
                    container = av.open(self.path)
                    audio_streams = [s for s in container.streams if s.type == "audio"]

                    if not audio_streams:
                        print(f"[AUDIO] No audio stream found in {os.path.basename(self.path)}")
                        return

                    audio_stream = audio_streams[0]
                    print(
                        f"[AUDIO] Audio stream found: codec={audio_stream.codec_context.name} "
                        f"rate={audio_stream.codec_context.sample_rate}"
                    )

                    resampler = av.audio.resampler.AudioResampler(
                        format="fltp",
                        layout="stereo",
                        rate=48000,
                    )

                    stream = self._open_output()
                    with self._lock:
                        self.stream = stream
                    stream.start()
                    print("[AUDIO] Playback started")

                    for packet in container.demux(audio_stream):
                        if self.stop_event.is_set():
                            break

                        for decoded in packet.decode():
                            if self.stop_event.is_set():
                                break

                            frames = resampler.resample(decoded)
                            if not isinstance(frames, list):
                                frames = [frames]

                            for audio_frame in frames:
                                if self.stop_event.is_set():
                                    break

                                arr = audio_frame.to_ndarray()

                                # PyAV planar audio is [channels, samples].
                                # sounddevice wants [samples, channels].
                                if arr.ndim == 1:
                                    arr = arr.reshape(-1, 1)
                                elif arr.ndim == 2:
                                    if arr.shape[0] == 2:
                                        arr = arr.T
                                    elif arr.shape[1] != 2:
                                        arr = arr.T

                                arr = np.ascontiguousarray(arr, dtype=np.float32)

                                if arr.size:
                                    # Safety: always provide exactly 2 channels.
                                    if arr.ndim == 2 and arr.shape[1] == 1:
                                        arr = np.repeat(arr, 2, axis=1)
                                    elif arr.ndim != 2 or arr.shape[1] != 2:
                                        continue
                                    stream.write(arr)

                    # End of MOV: loop while VIDEO remains selected.
                    if not self.stop_event.is_set():
                        try:
                            container.close()
                        except Exception:
                            pass
                        container = None

                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
                        stream = None

                        # Small yield before reopening the MOV.
                        time.sleep(0.01)
                        continue

                    break

                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[AUDIO] Playback error: {exc}")
                        print("[AUDIO] Audio thread stopped. VIDEO can still display normally.")
                    break

        finally:
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:
                pass

            with self._lock:
                self.stream = None

            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

            self.running = False
            print("[AUDIO] Playback stopped")

    def stop(self):
        self.stop_event.set()

        # Stop the active output stream immediately.  This also wakes the
        # audio thread if it is currently writing PCM samples.
        with self._lock:
            stream = self.stream

        if stream is not None:
            try:
                stream.abort()
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass

        self.running = False

    def close(self):
        self.stop()


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
            # VIDEO is a real member of the filter cycle.
            # The actual MOV frame is rendered by render_video_portal().
            "video": lambda roi: roi,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.is_3d_mode = False
        
        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )

        self.portal_video = None
        self.audio_controller = None

        # A pinch is latched until the fingers separate, so one gesture
        # advances exactly one filter.
        self.pinch_latched = False

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    @property
    def is_video_filter(self) -> bool:
        return self.current_filter_name == "video"

    def set_audio_controller(self, audio_controller) -> None:
        """Attach PortalAudio after it has been created in main()."""
        self.audio_controller = audio_controller

    def _sync_video_state(self) -> None:
        """Start/stop MOV audio strictly with the VIDEO filter."""
        if self.audio_controller is None:
            return

        if self.is_video_filter:
            if self.portal_video is not None:
                self.portal_video.start_time = time.perf_counter()
            self.audio_controller.start()
            print("[VIDEO] VIDEO filter selected - audio ON")
        else:
            self.audio_controller.stop()
            print(f"[VIDEO] {self.current_filter_name.upper()} selected - audio OFF")

    def cycle_filter(self, step: int = 1, from_pinch: bool = False) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)
        # VIDEO audio is tied to filter selection, not to a separate V key.
        self._sync_video_state()

    def handle_pinch(self, is_pinching: bool, now: float) -> None:
        """
        One pinch gesture = exactly one next-filter action.

        Keeping the pinch latched prevents MediaPipe from advancing through
        several filters while thumb and index remain together.
        """
        if is_pinching:
            if not self.pinch_latched:
                if now - self.last_switch_time >= self.cfg.filter_cooldown_sec:
                    self.cycle_filter(1, from_pinch=True)
                    self.last_switch_time = now
                    self.pinch_latched = True
        else:
            self.pinch_latched = False

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        processed_roi = self.filters[filter_key](roi)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        return frame

    def render_video_portal(
        self,
        frame: np.ndarray,
        pts: List[Tuple[int, int]],
        video_frame: np.ndarray
    ) -> np.ndarray:
        """Warp the MOV frame into the hand-shaped portal."""
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        # Four points: perspective-map the whole video into the portal.
        if len(pts) == 4:
            target = GeometryUtils.sort_quad_clean(pts).astype(np.float32)
            sh, sw = video_frame.shape[:2]
            source = np.array(
                [[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]],
                dtype=np.float32
            )
            matrix = cv2.getPerspectiveTransform(source, target)
            warped = cv2.warpPerspective(
                video_frame, matrix,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            # Slightly soften the portal edge.
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            frame[:] = (
                frame.astype(np.float32) * (1.0 - alpha)
                + warped.astype(np.float32) * alpha
            ).astype(np.uint8)
        else:
            # Six-point 3D mode: fit the video to the polygon's bounding box.
            resized = cv2.resize(video_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            roi = frame[y:y+h, x:x+w]
            local_poly = poly - np.array([x, y])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [local_poly], 255)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            roi[:] = (
                roi.astype(np.float32) * (1.0 - alpha)
                + resized.astype(np.float32) * alpha
            ).astype(np.uint8)

        # White portal border, matching the reference look.
        cv2.polylines(
            frame, [poly], isClosed=True,
            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )
        return frame

    def resize_crop_aspect(self, frame: np.ndarray) -> np.ndarray:
        """Resize the camera image to the configured output without stretching."""
        target_w = self.cfg.frame_width
        target_h = self.cfg.frame_height

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        target_ratio = target_w / target_h
        current_ratio = w / h

        # Crop the excess dimension so the source and output have the same ratio.
        if current_ratio < target_ratio:
            # Source is too tall (e.g. 4:3 -> 16:9): crop top/bottom.
            new_h = max(1, int(w / target_ratio))
            y1 = max(0, (h - new_h) // 2)
            frame = frame[y1:y1 + new_h, :]
        elif current_ratio > target_ratio:
            # Source is too wide: crop left/right.
            new_w = max(1, int(h * target_ratio))
            x1 = max(0, (w - new_w) // 2)
            frame = frame[:, x1:x1 + new_w]

        return cv2.resize(
            frame,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        frame = self.resize_crop_aspect(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.detector.process(rgb)
        now = time.time()

        # Read the MOV frame AFTER pinch handling below.
        # This is important: when a pinch selects VIDEO, the new filter state
        # must be used on the same camera frame instead of waiting for another
        # frame.
        video_frame = None

        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                lm = hand_lm.landmark
                tips = [(int(lm[i].x * self.cfg.frame_width), int(lm[i].y * self.cfg.frame_height)) for i in [4, 8, 12, 16, 20]]
                all_hand_tips.append(tips)

                # Thumb + index pinch = NEXT FILTER.
                is_pinching = (
                    GeometryUtils.euclidean_dist(tips[0], tips[1])
                    < self.cfg.pinch_threshold_px
                )
                self.handle_pinch(is_pinching, now)

                if GeometryUtils.is_fist_closed(lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_dist_threshold_px):
                    fist_count += 1

            # Fetch the MOV after the pinch may have changed the active filter.
            if self.is_video_filter and self.portal_video is not None:
                video_frame = self.portal_video.get_frame()

            # Dual Fist Mode Switch
            if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
                self.is_3d_mode = not self.is_3d_mode
                self.last_mode_toggle = now

            if self.is_3d_mode:
                if len(all_hand_tips) == 2:
                    t1, t2 = all_hand_tips[0], all_hand_tips[1]
                    portal1 = [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]]
                    portal2 = [t1[2], t1[3], t1[4], t2[4], t2[3], t2[2]]
                    if self.is_video_filter and video_frame is not None:
                        frame = self.render_video_portal(frame, portal1, video_frame)
                        frame = self.render_video_portal(frame, portal2, video_frame)
                    else:
                        frame = self.render_portal(frame, portal1, self.current_filter_name)
                        frame = self.render_portal(frame, portal2, self.secondary_filter_name)
                elif len(all_hand_tips) == 1:
                    if self.is_video_filter and video_frame is not None:
                        frame = self.render_video_portal(frame, all_hand_tips[0], video_frame)
                    else:
                        frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
            else:
                if len(all_hand_tips) == 2:
                    corners = [all_hand_tips[0][0], all_hand_tips[0][1], all_hand_tips[1][0], all_hand_tips[1][1]]
                    if GeometryUtils.is_hand_rotated(corners[0], corners[1]) or GeometryUtils.is_hand_rotated(corners[2], corners[3]):
                        quad = GeometryUtils.sort_quad_bowtie(corners)
                        is_bowtie = True
                    else:
                        quad = GeometryUtils.sort_quad_clean(corners)
                    if self.is_video_filter and video_frame is not None:
                        frame = self.render_video_portal(frame, quad.tolist(), video_frame)
                    else:
                        frame = self.render_portal(frame, quad.tolist(), self.current_filter_name)
                elif len(all_hand_tips) == 1:
                    t = all_hand_tips[0]
                    portal = [t[0], t[1], t[2], t[4]]
                    if self.is_video_filter and video_frame is not None:
                        frame = self.render_video_portal(frame, portal, video_frame)
                    else:
                        frame = self.render_portal(frame, portal, self.current_filter_name)

        # HUD hidden: hand detection and controls still work normally.
        return frame

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool) -> None:
        mode_str = "3D Mesh" if self.is_3d_mode else ("2D Bowtie" if is_bowtie else "2D Quad")
        cv2.putText(frame, f"MODE: {mode_str} [Key 'C' / Dual Fist]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FILTER: {self.current_filter_name.upper()} [Pinch / Key 'N'/'P']", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def list_cameras(max_cameras: int = 15) -> list[tuple[int, str]]:
    """Find available Windows camera indexes."""
    available = []
    print("\nScanning for cameras...")

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append((index, f"Camera {index}"))

        cap.release()

    return available


def find_obs_virtual_camera(max_cameras: int = 15) -> int | None:
    """
    Automatically locate OBS Virtual Camera.

    OpenCV's DirectShow backend usually does not expose the camera's
    device name. We therefore test each available camera and look for
    the OBS Virtual Camera by opening it through the DirectShow device
    list when possible. If the name cannot be detected, this function
    returns None so the normal selector can be used safely.
    """
    try:
        # Try Windows DirectShow device enumeration through PowerShell.
        import subprocess

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.Name -match 'OBS Virtual Camera' } | "
                "Select-Object -ExpandProperty Name"
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if "OBS Virtual Camera" not in result.stdout:
            return None

    except Exception:
        return None

    # OBS is installed and its virtual camera device exists.
    # OpenCV doesn't reliably provide the DirectShow device index,
    # so test the indexes and return the first camera that opens.
    #
    # On most Windows installations OBS Virtual Camera is exposed
    # as one of the first available capture devices.
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                print(f"OBS Virtual Camera detected. Trying camera index {index}.")
                return index

    return None


def choose_camera() -> int | None:
    """Automatically select OBS Virtual Camera when it is installed."""
    obs_index = find_obs_virtual_camera()

    if obs_index is not None:
        print(f"\n[OBS] Automatically selected OBS Virtual Camera (index {obs_index})")
        return obs_index

    print("\n[OBS] OBS Virtual Camera was not detected.")
    print("Make sure OBS is open and click 'Start Virtual Camera'.")
    print("Falling back to manual camera selection.")

    cameras = list_cameras()

    if not cameras:
        print("[ERROR] No cameras were found!")
        return None

    print("\nAvailable cameras:")
    for number, (index, name) in enumerate(cameras, start=1):
        print(f"  {number}. {name} (index {index})")

    while True:
        choice = input(f"\nChoose camera [1-{len(cameras)}]: ").strip()

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(cameras):
                selected = cameras[choice_num - 1][0]
                print(f"Using camera index {selected}")
                return selected
        except ValueError:
            pass

        print("Invalid choice. Please enter one of the numbers above.")

def main() -> None:
    cfg = PipelineConfig()

    selected_camera = choose_camera()
    if selected_camera is None:
        return

    cfg.cam_index = selected_camera
    processor = PortalProcessor(cfg)
    print("[KEYS] Pinch = next filter | N/P = previous/next filter | C = 3D mode | S = screenshot | Q = quit")
    print("[FILTERS] Dual Tone -> Thermal -> Sketch -> Pixelate -> Glitch -> Invert -> Red -> Edge -> Blur -> Cartoon -> Rainbow -> VIDEO -> Dual Tone")
    print("[VIDEO] VIDEO is a normal filter. Pinch to select it; pinch again to leave it.")

    # The MOV becomes the content inside the hand portal.
    video_path = cfg.portal_video_path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(video_path):
        video_path = os.path.join(script_dir, video_path)

    # Accept both the original filename and the uploaded "(1)" filename.
    # This prevents VIDEO from silently disappearing just because Windows
    # renamed a duplicate upload.
    if not os.path.isfile(video_path):
        candidates = [
            os.path.join(script_dir, "IMG_7343(1).MOV"),
            os.path.join(script_dir, "IMG_7343.MOV"),
        ]
        video_path = next((p for p in candidates if os.path.isfile(p)), video_path)

    portal_audio = None
    try:
        processor.portal_video = VideoPortal(video_path)
        portal_audio = PortalAudio(video_path)
        processor.set_audio_controller(portal_audio)
        # IMPORTANT: audio does NOT start here.
        # It starts only when a pinch selects VIDEO.
        print(f"[VIDEO] Portal video loaded: {video_path}")
        if PYTHON_AUDIO_AVAILABLE:
            print("[AUDIO] Using PyAV + sounddevice (no ffmpeg/ffplay executable needed).")
        else:
            print("[AUDIO] Audio disabled. Install with: python -m pip install av sounddevice")
    except Exception as exc:
        print(f"[VIDEO] Could not load portal video: {exc}")
        print("       Put IMG_7343(1).MOV or IMG_7343.MOV in the same folder as this Python file.")

    # CAP_DSHOW helps camera selection work more reliably on Windows.
    cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[ERROR] Could not open Camera {cfg.cam_index}!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        out_frame = processor.process_frame(frame)
        cv2.imshow("RetroLens Engine", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            processor.is_3d_mode = not processor.is_3d_mode
        elif key == ord("n"):
            processor.cycle_filter(1)
        elif key == ord("p"):
            processor.cycle_filter(-1)
        elif key == ord("s"):
            cv2.imwrite(f"cap_{int(time.time())}.png", out_frame)

    cap.release()
    if processor.portal_video is not None:
        processor.portal_video.close()
    if portal_audio is not None:
        portal_audio.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()