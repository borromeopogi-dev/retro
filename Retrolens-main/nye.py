import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# -----------------------------
# MediaPipe
# -----------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("\nMediaPipe is missing.")
    print("Install it with:")
    print("  python -m pip install mediapipe opencv-python numpy\n")
    sys.exit(1)


# -----------------------------
# Configuration
# -----------------------------
CAMERA_INDEX = 0

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

MODEL_NAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    MODEL_NAME
)

# Rain overlay image (green-screen rain image).
RAIN_IMAGE_NAME = "rain_overlay.png"
RAIN_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    RAIN_IMAGE_NAME
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "PORTAL CUBE - Finger Frame V2"

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# -----------------------------
# Download model
# -----------------------------
def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return

    print("MediaPipe hand model not found.")
    print("Downloading hand_landmarker.task ...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print("\nCould not download the MediaPipe model.")
        print("Download this file manually:")
        print(MODEL_URL)
        print("\nSave it next to finger_portal_v2.py as:")
        print("hand_landmarker.task")
        print("\nError:", exc)
        sys.exit(1)

    print("Model downloaded successfully.")


# -----------------------------
# Geometry
# -----------------------------
def pxy(landmark, width, height):
    return np.array(
        [landmark.x * width, landmark.y * height],
        dtype=np.float32
    )


def polygon_order(points):
    """
    Order four points clockwise around their center.
    This handles tilted/perspective frames.
    """
    pts = np.asarray(points, dtype=np.float32)

    center = np.mean(pts, axis=0)
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    return pts[np.argsort(angles)]


def polygon_area(points):
    return abs(
        cv2.contourArea(
            np.asarray(points, dtype=np.float32)
        )
    )


def smooth_points(old, new, amount=0.25):
    if old is None:
        return new.astype(np.float32)

    return (
        old * (1.0 - amount)
        + new.astype(np.float32) * amount
    )


def get_hand_points(result, width, height):
    """
    Uses the thumb tip (4) and index tip (8) of each hand.

    Four points:
        left hand  -> thumb + index
        right hand -> thumb + index

    The final points are sorted around the polygon.
    """
    if not result.hand_landmarks:
        return None

    if len(result.hand_landmarks) < 2:
        return None

    hands = []

    for landmarks in result.hand_landmarks[:2]:
        wrist = pxy(landmarks[0], width, height)
        thumb = pxy(landmarks[4], width, height)
        index = pxy(landmarks[8], width, height)

        # Reject a hand that is extremely small.
        palm = np.linalg.norm(
            pxy(landmarks[9], width, height) - wrist
        )

        if palm < 15:
            continue

        hands.append({
            "wrist": wrist,
            "thumb": thumb,
            "index": index,
        })

    if len(hands) != 2:
        return None

    # Sort hands left -> right.
    hands.sort(key=lambda h: h["wrist"][0])

    raw = [
        hands[0]["thumb"],
        hands[0]["index"],
        hands[1]["index"],
        hands[1]["thumb"],
    ]

    points = polygon_order(raw)

    # Frame must be reasonably large.
    area = polygon_area(points)

    if area < width * height * 0.025:
        return None

    if area > width * height * 0.90:
        return None

    return points


# -----------------------------
# Image effects
# -----------------------------
def effect_normal(frame, t):
    return frame.copy()


def effect_negative(frame, t):
    return cv2.bitwise_not(frame)


def effect_blue_thermal(frame, t):
    """
    Similar to the strong blue/cyan face look in the reference.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Animated brightness pulse.
    pulse = 15.0 * np.sin(t * 3.0)
    gray = np.clip(gray.astype(np.float32) + pulse, 0, 255)
    gray = gray.astype(np.uint8)

    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Push the result toward cyan/blue.
    b, g, r = cv2.split(thermal)

    b = cv2.add(b, 40)
    g = cv2.add(g, 15)

    return cv2.merge([b, g, r])


def effect_rgb_split(frame, t):
    """
    Chromatic aberration / RGB separation.
    """
    shift = int(4 + 3 * abs(np.sin(t * 5.0)))

    b, g, r = cv2.split(frame)

    b = np.roll(b, -shift, axis=1)
    r = np.roll(r, shift, axis=1)

    result = cv2.merge([b, g, r])

    # Add a small contrast boost.
    result = cv2.convertScaleAbs(result, alpha=1.12, beta=0)

    return result


def effect_glitch(frame, t):
    """
    Fast horizontal glitch bands.
    """
    result = frame.copy()
    h, w = result.shape[:2]

    rng = np.random.default_rng(int(t * 20))

    number_of_bands = 7

    for _ in range(number_of_bands):
        y = int(rng.integers(0, max(1, h - 4)))
        band_h = int(rng.integers(2, max(3, h // 18)))

        shift = int(rng.integers(-w // 10, w // 10))

        y2 = min(h, y + band_h)

        result[y:y2] = np.roll(
            result[y:y2],
            shift,
            axis=1
        )

    return result


def effect_mirror(frame, t):
    """
    Mirrored portal look.
    """
    h, w = frame.shape[:2]

    left = frame[:, :w // 2]
    right = cv2.flip(left, 1)

    result = np.hstack([left, right])

    return result


def effect_scanlines(frame, t):
    result = frame.copy()

    # Vectorized instead of looping through every scanline.
    result[::3] = (
        result[::3].astype(np.float32) * 0.72
    ).astype(np.uint8)

    b, g, r = cv2.split(result)
    r = np.roll(r, 1, axis=1)

    return cv2.merge([b, g, r])

def effect_portal(frame, t):
    """Fast portal warp. Works on a small internal image."""
    h, w = frame.shape[:2]

    if h < 10 or w < 10:
        return frame

    scale = 0.50
    sw = max(16, int(w * scale))
    sh = max(16, int(h * scale))

    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)

    cx = sw * 0.5
    cy = sh * 0.5

    dx = xx - cx
    dy = yy - cy

    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = max(1.0, np.sqrt(cx * cx + cy * cy))

    ripple = np.sin(radius * 0.10 - t * 5.0)
    strength = 4.0 * (radius / max_radius) ** 1.35

    angle = 0.035 * np.sin(t * 1.7) * (radius / max_radius)

    ca = np.cos(angle)
    sa = np.sin(angle)

    sx = dx * ca - dy * sa
    sy = dx * sa + dy * ca

    sx += (dx / (radius + 1.0)) * ripple * strength
    sy += (dy / (radius + 1.0)) * ripple * strength

    map_x = np.clip(cx + sx, 0, sw - 1).astype(np.float32)
    map_y = np.clip(cy + sy, 0, sh - 1).astype(np.float32)

    warped = cv2.remap(
        small, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    warped = cv2.resize(
        warped, (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    shift = max(1, min(2, w // 250))
    b, g, r = cv2.split(warped)

    b = np.roll(b, -shift, axis=1)
    r = np.roll(r, shift, axis=1)

    result = cv2.merge([b, g, r])

    return cv2.convertScaleAbs(
        result,
        alpha=1.12,
        beta=-6
    )

def effect_rain_overlay(frame, t):
    """
    Uses rain_overlay.png as a green-screen rain overlay.
    The bright green background is made transparent so only
    the rain streaks are shown over the webcam image.
    """
    if not os.path.isfile(RAIN_IMAGE_PATH):
        return frame

    overlay = cv2.imread(RAIN_IMAGE_PATH, cv2.IMREAD_COLOR)
    if overlay is None:
        return frame

    h, w = frame.shape[:2]
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

    # The uploaded image has a bright green background.
    # Keep the light/white rain streaks and remove the green.
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)

    # Green-screen mask.
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 150, 120], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8)
    )

    rain_mask = cv2.bitwise_not(green_mask)

    # Keep only brighter pixels so the green background doesn't leak through.
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.inRange(gray, 120, 255)
    rain_mask = cv2.bitwise_and(rain_mask, bright_mask)

    # Slightly soften the mask for a natural overlay.
    rain_mask = cv2.GaussianBlur(rain_mask, (3, 3), 0)

    alpha = rain_mask.astype(np.float32)[:, :, None] / 255.0

    # Blend the rain over the webcam frame.
    result = (
        overlay.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


EFFECTS = {
    1: ("NORMAL", effect_normal),
    2: ("NEGATIVE", effect_negative),
    3: ("BLUE THERMAL", effect_blue_thermal),
    4: ("RGB SPLIT", effect_rgb_split),
    5: ("GLITCH", effect_glitch),
    6: ("MIRROR", effect_mirror),
    7: ("SCANLINES", effect_scanlines),
    8: ("PORTAL WARP", effect_portal),
    9: ("RAIN OVERLAY", effect_rain_overlay),
}


# -----------------------------
# Apply effect only inside quad
# -----------------------------
def composite_quad(frame, points, effect_function, effect_time):
    """Process only the bounding box of the finger frame."""
    h, w = frame.shape[:2]
    pts = points.astype(np.int32)

    x0 = max(0, int(np.min(pts[:, 0])) - 6)
    y0 = max(0, int(np.min(pts[:, 1])) - 6)
    x1 = min(w, int(np.max(pts[:, 0])) + 7)
    y1 = min(h, int(np.max(pts[:, 1])) + 7)

    if x1 <= x0 or y1 <= y0:
        return frame

    roi = frame[y0:y1, x0:x1]

    local = pts.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0

    processed = effect_function(roi, effect_time)

    mask = np.zeros(
        (y1 - y0, x1 - x0),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(mask, local, 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    alpha = mask.astype(np.float32)[:, :, None] / 255.0

    roi[:] = np.clip(
        processed.astype(np.float32) * alpha
        + roi.astype(np.float32) * (1.0 - alpha),
        0, 255
    ).astype(np.uint8)

    return frame

# -----------------------------
# Wireframe UI
# -----------------------------
def draw_corner(frame, x, y, size=15):
    cv2.circle(
        frame,
        (int(x), int(y)),
        5,
        GREEN,
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (int(x), int(y)),
        9,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # Small L-shaped corner marks.
    cv2.line(
        frame,
        (int(x - size), int(y)),
        (int(x - 3), int(y)),
        GREEN,
        2
    )

    cv2.line(
        frame,
        (int(x + 3), int(y)),
        (int(x + size), int(y)),
        GREEN,
        2
    )

    cv2.line(
        frame,
        (int(x), int(y - size)),
        (int(x), int(y - 3)),
        GREEN,
        2
    )

    cv2.line(
        frame,
        (int(x), int(y + 3)),
        (int(x), int(y + size)),
        GREEN,
        2
    )


def draw_wireframe(frame, points, effect_name, show_labels=True):
    if points is None:
        return

    pts = points.astype(np.int32)

    # Main green quadrilateral.
    cv2.polylines(
        frame,
        [pts],
        True,
        GREEN,
        2,
        cv2.LINE_AA
    )

    # Bright inner line.
    center = np.mean(pts, axis=0)

    inner = (
        center
        + (pts.astype(np.float32) - center) * 0.96
    ).astype(np.int32)

    cv2.polylines(
        frame,
        [inner],
        True,
        (0, 160, 0),
        1,
        cv2.LINE_AA
    )

    for x, y in pts:
        draw_corner(frame, x, y)

    if not show_labels:
        return

    x = int(np.min(pts[:, 0]))
    y = max(28, int(np.min(pts[:, 1])) - 12)

    # Label similar to the reference video.
    label = f"PORTAL CUBE - {effect_name}"

    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        GREEN,
        2,
        cv2.LINE_AA
    )


def draw_hud(frame, effect_name, fps, auto_mode, show_labels):
    h, w = frame.shape[:2]

    # Top-left diagnostic text.
    lines = [
        "PORTAL CUBE / FINGER FRAME V2",
        f"EFFECT: {effect_name}",
        f"FPS: {fps:5.1f}",
        f"AUTO: {'ON' if auto_mode else 'OFF'}",
    ]

    y = 28

    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GREEN,
            1,
            cv2.LINE_AA
        )
        y += 20

    if show_labels:
        cv2.putText(
            frame,
            "BOTH HANDS: THUMB + INDEX",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA
        )

    # Thin green border around the webcam view.
    cv2.rectangle(
        frame,
        (3, 3),
        (w - 4, h - 4),
        GREEN,
        1
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_model()

    # MediaPipe Tasks configuration.
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    detector = vision.HandLandmarker.create_from_options(
        options
    )

    # Try Windows DirectShow first.
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("\nERROR: Webcam could not be opened.")
        print("Try CAMERA_INDEX = 1 near the top of the script.")
        print("Also close Zoom, Discord, OBS, Teams, etc.")
        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # Start with the portal effect because it is closest
    # to the later blue/warped portions of the reference.
    effect_id = 8

    auto_mode = False
    show_labels = True

    # Auto-cycle timing.
    AUTO_SECONDS = 3.2
    auto_started = time.monotonic()

    # Smoothed quadrilateral.
    smoothed = None

    # If tracking disappears for a tiny moment, keep the
    # previous frame for this long.
    lost_since = None
    LOST_GRACE = 0.35

    # FPS.
    previous_time = time.perf_counter()
    fps = 0.0

    # MediaPipe video timestamps must increase.
    timestamp_ms = 0
    detect_counter = 0
    last_result = None
    DETECT_EVERY = 2

    print("\n======================================")
    print(" PORTAL CUBE / FINGER FRAME V2")
    print("======================================")
    print("Make a frame using BOTH hands.")
    print("Use your thumb + index finger on each hand.")
    print("")
    print("1 Normal")
    print("2 Negative")
    print("3 Blue Thermal")
    print("4 RGB Split")
    print("5 Glitch")
    print("6 Mirror")
    print("7 Scanlines")
    print("8 Portal Warp")
    print("9 Rain Overlay")
    print("0 Auto Cycle")
    print("SPACE Next Effect")
    print("A Toggle Auto")
    print("F Toggle Labels")
    print("P Performance Mode: 640x360")
    print("Q / ESC Quit")
    print("======================================\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Webcam frame could not be read.")
            break

        # Selfie mirror.
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # -------------------------
        # Auto effect
        # -------------------------
        if auto_mode:
            elapsed = time.monotonic() - auto_started

            if elapsed >= AUTO_SECONDS:
                effect_id += 1

                if effect_id > 9:
                    effect_id = 2

                auto_started = time.monotonic()

        effect_name, effect_function = EFFECTS[effect_id]

        # -------------------------
        # MediaPipe hand detection
        # -------------------------
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms += 33

        detect_counter += 1

        if detect_counter >= DETECT_EVERY or last_result is None:
            last_result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )
            detect_counter = 0

        current_points = get_hand_points(
            last_result,
            width,
            height
        )

        if current_points is not None:
            smoothed = smooth_points(
                smoothed,
                current_points,
                amount=0.30
            )

            lost_since = None

        elif lost_since is None:
            lost_since = time.monotonic()

        # Drop the old frame after the grace period.
        if (
            lost_since is not None
            and time.monotonic() - lost_since > LOST_GRACE
        ):
            smoothed = None

        points = smoothed.astype(np.int32) if smoothed is not None else None

        # -------------------------
        # Apply portal effect
        # -------------------------
        if points is not None:
            frame = composite_quad(
                frame,
                points,
                effect_function,
                time.monotonic()
            )

            # Wireframe is drawn AFTER the effect.
            draw_wireframe(
                frame,
                points,
                effect_name,
                show_labels
            )

        # -------------------------
        # FPS
        # -------------------------
        now = time.perf_counter()
        dt = now - previous_time
        previous_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            fps = (
                instant_fps
                if fps == 0
                else fps * 0.90 + instant_fps * 0.10
            )

        draw_hud(
            frame,
            effect_name,
            fps,
            auto_mode,
            show_labels
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit.
        if key in (27, ord("q"), ord("Q")):
            break

        # Number effects.
        if ord("1") <= key <= ord("9"):
            effect_id = int(chr(key))
            auto_started = time.monotonic()

        # 0 = auto cycle.
        elif key == ord("0"):
            auto_mode = True
            auto_started = time.monotonic()

        # Space = next effect.
        elif key == 32:
            effect_id += 1

            if effect_id > 9:
                effect_id = 1

            auto_started = time.monotonic()

        # A = auto toggle.
        elif key in (ord("a"), ord("A")):
            auto_mode = not auto_mode
            auto_started = time.monotonic()

        # F = labels toggle.
        elif key in (ord("f"), ord("F")):
            show_labels = not show_labels

        elif key in (ord("p"), ord("P")):
            # Lower resolution gives a large FPS improvement.
            current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            if current_width > 700:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()