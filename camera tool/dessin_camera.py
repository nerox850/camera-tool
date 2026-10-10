"""Construire des volumes 3D filaires et reconnaître localement les visages.

Placez plusieurs photos de chaque personne dans visages/Nom/ (ex. visages/Alice/),
puis lancez le programme avec ./lancer_linux.sh ou lancer_windows.bat.
Cliquez sur une forme de la palette pour la placer, puis pincez-la pour la déplacer.
Le dessin libre reste disponible en faisant glisser la souris ou en pointant
avec l'index, puis cliquez sur Finaliser pour l'ajouter.
Le suivi affiche le squelette du corps et des deux mains. Appuyez sur Q pour
quitter.
"""

import sys
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "Dépendances manquantes. Installez-les avec : "
        "python -m pip install -r requirements.txt"
    ) from error

if not hasattr(mp, "solutions"):
    raise SystemExit(
        "Version de MediaPipe incompatible. Relancez le programme avec "
        "lancer_linux.sh ou lancer_windows.bat pour installer la version requise."
    )

WIDTH, HEIGHT = 960, 720
RED = (0, 0, 255)
FACE_SIZE = (160, 160)
FACE_CONFIDENCE_LIMIT = 75
FACE_EDGES = {
    tuple(sorted(edge))
    for edge in mp.solutions.face_mesh.FACEMESH_TESSELATION
}
POSE_EDGES = mp.solutions.holistic.POSE_CONNECTIONS
OBJECT_TEMPLATES = {}
PRESET_LABELS = {
    "cube": "Cube",
    "sphere": "Sphere",
    "pyramide": "Pyramide",
    "prisme": "Prisme",
    "coeur": "Coeur",
    "etoile": "Etoile",
    "fleche": "Fleche",
    "nuage": "Nuage",
}


def build_object_templates():
    def sampled_polygon(vertices):
        points = []
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            for fraction in np.linspace(0, 1, 12, endpoint=False):
                points.append(
                    (
                        round(start[0] * (1 - fraction) + end[0] * fraction),
                        round(start[1] * (1 - fraction) + end[1] * fraction),
                    )
                )
        return points

    star = []
    for index in range(10):
        radius = 100 if index % 2 == 0 else 45
        angle = -np.pi / 2 + index * np.pi / 5
        star.append(
            (150 + int(radius * np.cos(angle)), 150 + int(radius * np.sin(angle)))
        )

    heart = []
    for angle in np.linspace(0, 2 * np.pi, 160, endpoint=False):
        x = 16 * np.sin(angle) ** 3
        y = (
            13 * np.cos(angle)
            - 5 * np.cos(2 * angle)
            - 2 * np.cos(3 * angle)
            - np.cos(4 * angle)
        )
        heart.append((150 + int(5 * x), 150 - int(5 * y)))

    arrow = sampled_polygon(
        [(30, 115), (230, 115), (230, 55), (340, 145),
         (230, 235), (230, 175), (30, 175)]
    )
    cloud = sampled_polygon(
        [
            (120, 280), (125, 230), (165, 215), (180, 160),
            (225, 145), (270, 175), (310, 155), (355, 185),
            (375, 230), (400, 235), (410, 280),
        ]
    )

    templates = {"Coeur": heart, "Etoile": star, "Fleche": arrow, "Nuage": cloud}
    return {
        name: np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        for name, points in templates.items()
    }


OBJECT_TEMPLATES = build_object_templates()


def load_face_recognizer(face_dir):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Impossible de charger le détecteur de visage : {cascade_path}")
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("OpenCV contrib est requis. Relancez le lanceur pour réparer l'installation.")

    face_dir.mkdir(exist_ok=True)
    images = []
    labels = []
    names = {}
    label = 0
    valid_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

    for person_dir in sorted(path for path in face_dir.iterdir() if path.is_dir()):
        person_images = sorted(
            path for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_suffixes
        )
        if not person_images:
            continue

        names[label] = person_dir.name
        for image_path in person_images:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Photo illisible, ignorée : {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) != 1:
                print(
                    f"Photo ignorée ({len(faces)} visage(s), un seul attendu) : "
                    f"{image_path}"
                )
                continue
            x, y, width, height = faces[0]
            crop = gray[y:y + height, x:x + width]
            images.append(cv2.equalizeHist(cv2.resize(crop, FACE_SIZE)))
            labels.append(label)
        label += 1

    if not images:
        print(
            f"Aucune photo de référence valide dans {face_dir}/. "
            "Créez un sous-dossier par personne et ajoutez-y ses photos."
        )
        return None, {}, detector

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    print(
        f"Reconnaissance faciale locale activée : {len(images)} photo(s), "
        f"{len(set(labels))} personne(s)."
    )
    return recognizer, names, detector


def identify_faces(frame, recognizer, names, detector):
    if recognizer is None:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
    )
    for x, y, width, height in faces:
        crop = cv2.equalizeHist(
            cv2.resize(gray[y:y + height, x:x + width], FACE_SIZE)
        )
        label, confidence = recognizer.predict(crop)
        name = names.get(label, "Inconnu") if confidence <= FACE_CONFIDENCE_LIMIT else "Inconnu"
        cv2.rectangle(frame, (x, y), (x + width, y + height), RED, 1)
        cv2.putText(
            frame,
            f"{name} ({confidence:.0f})",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            RED,
            2,
            cv2.LINE_AA,
        )


def draw_face_wireframe(frame, face_results):
    face = face_results.face_landmarks
    if face is None:
        return
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(point.x * width, 0, width - 1)),
            int(np.clip(point.y * height, 0, height - 1)),
        )
        for point in face.landmark
    ]
    for start, end in FACE_EDGES:
        cv2.line(frame, points[start], points[end], RED, 1, cv2.LINE_AA)


def draw_pose_skeleton(frame, pose_landmarks):
    if pose_landmarks is None:
        return
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(point.x * width, 0, width - 1)),
            int(np.clip(point.y * height, 0, height - 1)),
        )
        for point in pose_landmarks.landmark
    ]
    for start, end in POSE_EDGES:
        first, second = pose_landmarks.landmark[start], pose_landmarks.landmark[end]
        if first.visibility < 0.45 or second.visibility < 0.45:
            continue
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2, cv2.LINE_AA)
    for index, landmark in enumerate(pose_landmarks.landmark):
        if landmark.visibility >= 0.5:
            cv2.circle(frame, points[index], 3, (0, 255, 255), -1, cv2.LINE_AA)


def detect_fingers(points):
    wrist = np.asarray((points[0].x, points[0].y), dtype=float)
    fingers = {
        "Pouce": (4, 3),
        "Index": (8, 6),
        "Majeur": (12, 10),
        "Annulaire": (16, 14),
        "Auriculaire": (20, 18),
    }
    states = {}
    for name, (tip_index, joint_index) in fingers.items():
        tip = np.asarray((points[tip_index].x, points[tip_index].y), dtype=float)
        joint = np.asarray((points[joint_index].x, points[joint_index].y), dtype=float)
        tip_distance = np.linalg.norm(tip - wrist)
        joint_distance = np.linalg.norm(joint - wrist)
        states[name] = tip_distance > joint_distance * 1.08
    return states


def draw_hand_tracking(frame, hand_landmarks, finger_states):
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(point.x * width, 0, width - 1)),
            int(np.clip(point.y * height, 0, height - 1)),
        )
        for point in hand_landmarks.landmark
    ]
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2, cv2.LINE_AA)
    tip_indices = {
        "Pouce": 4,
        "Index": 8,
        "Majeur": 12,
        "Annulaire": 16,
        "Auriculaire": 20,
    }
    for name, index in tip_indices.items():
        color = (0, 220, 0) if finger_states[name] else (100, 100, 255)
        cv2.circle(frame, points[index], 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, points[index], 10, (255, 255, 255), 1, cv2.LINE_AA)


def extract_hand_gesture(frame, hand_landmarks, hand_name):
    if hand_landmarks is None:
        return None
    finger_states = detect_fingers(hand_landmarks.landmark)
    draw_hand_tracking(frame, hand_landmarks, finger_states)
    height, width = frame.shape[:2]
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    tip = (
        int(np.clip(index_tip.x * width, 0, width - 1)),
        int(np.clip(index_tip.y * height, 0, height - 1)),
    )
    thumb = (
        int(np.clip(thumb_tip.x * width, 0, width - 1)),
        int(np.clip(thumb_tip.y * height, 0, height - 1)),
    )
    pinching = np.hypot(tip[0] - thumb[0], tip[1] - thumb[1]) < 38
    cursor = ((tip[0] + thumb[0]) // 2, (tip[1] + thumb[1]) // 2)
    if pinching:
        cv2.circle(frame, cursor, 14, RED, 2, cv2.LINE_AA)
    return {
        "name": hand_name,
        "tip": tip,
        "cursor": cursor,
        "pinching": pinching,
        "finger_states": finger_states,
    }


def project_3d(point, center, focal_length):
    x, y, z = point
    yaw = np.deg2rad(32)
    pitch = np.deg2rad(-24)
    rotated_x = np.cos(yaw) * x + np.sin(yaw) * z
    rotated_z = -np.sin(yaw) * x + np.cos(yaw) * z
    rotated_y = np.cos(pitch) * y - np.sin(pitch) * rotated_z
    camera_z = np.sin(pitch) * y + np.cos(pitch) * rotated_z
    camera_distance = focal_length * 1.6
    scale = camera_distance / max(camera_distance - camera_z, 1)
    return (
        int(round(center[0] + rotated_x * scale)),
        int(round(center[1] + rotated_y * scale)),
    )


def draw_projected_lines(canvas, lines, center, focal_length):
    for points in lines:
        projected = np.asarray(
            [project_3d(point, center, focal_length) for point in points],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [projected], False, RED, 2, cv2.LINE_AA)


def draw_sphere(canvas, center, radius, focal_length):
    lines = []
    samples = np.linspace(0, 2 * np.pi, 65)
    for latitude in np.linspace(-np.pi / 2, np.pi / 2, 9)[1:-1]:
        lines.append(
            [
                (
                    radius * np.cos(latitude) * np.cos(longitude),
                    radius * np.sin(latitude),
                    radius * np.cos(latitude) * np.sin(longitude),
                )
                for longitude in samples
            ]
        )
    for longitude in np.linspace(0, np.pi, 9, endpoint=False):
        lines.append(
            [
                (
                    radius * np.cos(latitude) * np.cos(longitude),
                    radius * np.sin(latitude),
                    radius * np.cos(latitude) * np.sin(longitude),
                )
                for latitude in samples
            ]
        )
    draw_projected_lines(canvas, lines, center, focal_length)


def draw_pyramid(canvas, center, width, height, focal_length):
    depth = width * 0.62
    base_y = height * 0.34
    base = [
        (-width / 2, base_y, -depth / 2),
        (width / 2, base_y, -depth / 2),
        (width / 2, base_y, depth / 2),
        (-width / 2, base_y, depth / 2),
    ]
    apex = (0.0, -height / 2, 0.0)
    lines = [
        base + [base[0]],
        *[[apex, corner] for corner in base],
    ]
    draw_projected_lines(canvas, lines, center, focal_length)


def recognize_object(contour):
    x, y, width, height = cv2.boundingRect(contour)
    if width < 2 or height < 2:
        return None
    polygon = cv2.approxPolyDP(
        contour, max(2.0, cv2.arcLength(contour, True) * 0.018), True
    )[:, 0, :].astype(float)
    normalized = (polygon - np.asarray((x, y))) / np.asarray((width, height))
    if len(polygon) < 6:
        return None

    bottom = normalized[np.argmax(normalized[:, 1])]
    upper = normalized[normalized[:, 1] < 0.35]
    central_notch = normalized[
        (normalized[:, 0] >= 0.38)
        & (normalized[:, 0] <= 0.62)
        & (normalized[:, 1] >= 0.18)
        & (normalized[:, 1] <= 0.72)
    ]
    if (
        0.32 <= bottom[0] <= 0.68
        and len(upper[upper[:, 0] < 0.46]) > 0
        and len(upper[upper[:, 0] > 0.54]) > 0
        and any(
            np.min(upper[upper[:, 0] < 0.46, 1]) < notch[1] - 0.08
            and np.min(upper[upper[:, 0] > 0.54, 1]) < notch[1] - 0.08
            for notch in central_notch
        )
    ):
        return "Coeur"

    center = normalized.mean(axis=0)
    radii = np.linalg.norm(normalized - center, axis=1)
    if 8 <= len(radii) <= 12 and len(radii) % 2 == 0:
        alternating_depth = abs(
            float(np.mean(radii[::2]) - np.mean(radii[1::2]))
        ) / max(float(np.mean(radii)), 1e-6)
        if alternating_depth >= 0.32:
            return "Etoile"

    def normalize_contour(shape_contour):
        bounds = cv2.boundingRect(shape_contour)
        x0, y0, shape_width, shape_height = bounds
        points = shape_contour[:, 0, :].astype(np.float32)
        points[:, 0] = (points[:, 0] - x0) / max(shape_width, 1) * 200
        points[:, 1] = (points[:, 1] - y0) / max(shape_height, 1) * 200
        return np.rint(points).astype(np.int32).reshape((-1, 1, 2))

    normalized_contour = normalize_contour(contour)
    polygon_size = len(polygon)
    distances = sorted(
        (
            cv2.matchShapes(
                normalized_contour,
                normalize_contour(template),
                cv2.CONTOURS_MATCH_I1,
                0,
            ),
            name,
        )
        for name, template in OBJECT_TEMPLATES.items()
        if (
            name == "Fleche" and 6 <= polygon_size <= 8
            or name == "Nuage" and polygon_size >= 9
        )
    )
    if not distances:
        return None
    distance, name = distances[0]
    if distance <= {"Fleche": 0.08, "Nuage": 0.03}[name]:
        return name
    return None


def draw_wireframe_shape(canvas, points, closed):
    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    if len(contour) < 2:
        return "Trait trop court"

    x, y, width, height = cv2.boundingRect(contour)
    diagonal = float(np.hypot(width, height))
    is_closed = closed and len(contour) >= 4
    if is_closed and diagonal > 0:
        is_closed = np.linalg.norm(contour[0, 0] - contour[-1, 0]) <= max(
            24, diagonal * 0.18
        )

    perimeter = cv2.arcLength(contour, is_closed)
    if perimeter == 0:
        return "Trait trop court"
    polygon = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.018), is_closed)

    if not is_closed or len(polygon) < 3:
        cv2.polylines(canvas, [polygon], False, RED, 3, cv2.LINE_AA)
        return "Ligne géométrique"

    polygon_points = polygon[:, 0, :].astype(float)
    area = abs(cv2.contourArea(polygon))
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
    aspect_ratio = width / max(height, 1)
    center = (x + width // 2, y + height // 2)
    focal_length = max(width, height, 1) * 3.5
    object_name = recognize_object(contour) if is_closed else None
    if object_name is not None:
        label = object_name
    elif len(polygon) >= 7 and 0.72 <= aspect_ratio <= 1.4 and circularity >= 0.58:
        draw_sphere(canvas, center, max(min(width, height) / 2, 1), focal_length)
        return "Sphère 3D filaire"
    else:
        sides = len(polygon_points)
        labels = {3: "Prisme triangulaire", 5: "Prisme pentagonal", 6: "Prisme hexagonal"}
        label = labels.get(sides, "Forme libre")
        if sides == 4:
            vectors = np.roll(polygon_points, -1, axis=0) - polygon_points
            right_angles = all(
                abs(np.dot(vectors[index - 1], vectors[index]))
                <= 0.35
                * np.linalg.norm(vectors[index - 1])
                * np.linalg.norm(vectors[index])
                for index in range(4)
            )
            if right_angles:
                ratio = width / max(height, 1)
                label = (
                    "Cube"
                    if 0.85 <= ratio <= 1.15
                    else "Parallélépipède rectangle"
                )

    depth = max(24.0, min(width, height) * 0.65)
    local_points = polygon_points - np.asarray(center, dtype=float)
    front = [(point[0], point[1], -depth / 2) for point in local_points]
    back = [(point[0], point[1], depth / 2) for point in local_points]
    lines = [
        front + [front[0]],
        back + [back[0]],
        *[[front[index], back[index]] for index in range(len(front))],
    ]
    draw_projected_lines(canvas, lines, center, focal_length)
    return f"{label} 3D filaire"


def mouse_callback(event, x, y, _flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_button = next(
            (
                name
                for name, (x1, y1, x2, y2) in state["buttons"].items()
                if x1 <= x <= x2 and y1 <= y <= y2
            ),
            None,
        )
        state["action"] = clicked_button
        state["mouse_drawing"] = clicked_button is None
        state["mouse_points"] = [(x, y)] if clicked_button is None else []
    elif event == cv2.EVENT_MOUSEMOVE and state["mouse_drawing"]:
        if _flags & cv2.EVENT_FLAG_LBUTTON:
            point = (x, y)
            if (
                not state["mouse_points"]
                or np.linalg.norm(
                    np.asarray(point) - state["mouse_points"][-1]
                ) >= 3
            ):
                state["mouse_points"].append(point)
    elif event == cv2.EVENT_LBUTTONUP and state["mouse_drawing"]:
        point = (x, y)
        if (
            not state["mouse_points"]
            or np.linalg.norm(
                np.asarray(point) - state["mouse_points"][-1]
            ) >= 2
        ):
            state["mouse_points"].append(point)
        state["completed_stroke"] = state["mouse_points"]
        state["mouse_drawing"] = False


def smooth_point(previous, current, amount=0.38):
    if previous is None:
        return tuple(current)
    smoothed = (
        np.asarray(previous, dtype=float) * (1 - amount)
        + np.asarray(current, dtype=float) * amount
    )
    return tuple(np.rint(smoothed).astype(int))


def finalize_pending_shape(scene, pending_shape):
    if pending_shape is None:
        return False
    scene.append(pending_shape)
    return True


def make_preset_shape(name, center):
    size = 116
    if name == "sphere":
        points = [
            (
                center[0] + int(size * 0.46 * np.cos(angle)),
                center[1] + int(size * 0.46 * np.sin(angle)),
            )
            for angle in np.linspace(0, 2 * np.pi, 65)[:-1]
        ]
    elif name == "cube":
        half = size // 2
        points = [
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] - half),
            (center[0] + half, center[1] + half),
            (center[0] - half, center[1] + half),
        ]
        points.append(points[0])
    elif name in {"prisme", "pyramide"}:
        half = size // 2
        points = [
            (center[0], center[1] - half),
            (center[0] + half, center[1] + half),
            (center[0] - half, center[1] + half),
        ]
        points.append(points[0])
    else:
        contour = OBJECT_TEMPLATES[PRESET_LABELS[name]][:, 0, :].astype(float)
        minimum = contour.min(axis=0)
        maximum = contour.max(axis=0)
        extent = np.maximum(maximum - minimum, 1)
        scale = np.asarray((size, size * 0.82)) / extent
        points = np.rint(
            (contour - (minimum + maximum) / 2) * scale
            + np.asarray(center)
        ).astype(int).tolist()
    if np.linalg.norm(np.asarray(points[0]) - np.asarray(points[-1])) > 20:
        points.append(points[0])
    return {
        "points": [list(map(int, point)) for point in points],
        "offset": [0, 0],
        "closed": True,
        "preset": name,
    }


def hit_test_shape(shape, point, padding=24):
    points = np.asarray(shape["points"], dtype=np.int32).copy()
    offset = np.asarray(shape["offset"], dtype=np.int32)
    points += offset
    left, top = points.min(axis=0) - padding
    right, bottom = points.max(axis=0) + padding
    return left <= point[0] <= right and top <= point[1] <= bottom


def render_shape(canvas, shape):
    offset = np.asarray(shape["offset"], dtype=np.int32)
    points = np.asarray(shape["points"], dtype=np.int32) + offset
    if shape.get("preset") == "pyramide":
        x, y, width, height = cv2.boundingRect(points)
        center = (x + width // 2, y + height // 2)
        draw_pyramid(canvas, center, width, height, max(width, height) * 3.5)
        return "Pyramide 3D filaire"
    return draw_wireframe_shape(canvas, points, shape["closed"])


def main():
    face_dir = Path(__file__).resolve().parent / "visages"
    recognizer, names, face_detector = load_face_recognizer(face_dir)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Caméra inaccessible. Vérifiez qu'elle est branchée et autorisée.")
        camera.release()
        return 1

    scene = []
    pending_shape = None
    drawing = False
    stroke_points = []
    message = ""
    held_button = None
    hold_count = 0
    button_fired = False
    active_drag = None
    drag_offset = np.zeros(2, dtype=np.int32)
    buttons = {
        "effacer": (8, 14, 88, 58),
        "cube": (94, 14, 174, 58),
        "sphere": (180, 14, 262, 58),
        "pyramide": (268, 14, 366, 58),
        "prisme": (372, 14, 452, 58),
        "coeur": (458, 14, 535, 58),
        "etoile": (541, 14, 624, 58),
        "fleche": (630, 14, 713, 58),
        "nuage": (719, 14, 800, 58),
        "finaliser": (806, 14, 950, 58),
    }
    button_labels = {
        "effacer": "Effacer",
        "cube": "Cube",
        "sphere": "Sphere",
        "pyramide": "Pyramide",
        "prisme": "Prisme",
        "coeur": "Coeur",
        "etoile": "Etoile",
        "fleche": "Fleche",
        "nuage": "Nuage",
        "finaliser": "Ajouter dessin",
    }
    mouse_state = {
        "buttons": buttons,
        "action": None,
        "mouse_drawing": False,
        "mouse_points": [],
        "completed_stroke": None,
    }
    smoothed_tip = None
    smoothed_cursor = None
    preset_count = 0
    window_name = "Dessin par gestes"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, mouse_state)

    def activate_button(name):
        nonlocal pending_shape, message, preset_count
        if name == "effacer":
            scene.clear()
            pending_shape = None
            preset_count = 0
            message = "Scène effacée"
        elif name in PRESET_LABELS:
            column = preset_count % 5
            row = preset_count // 5
            center = (220 + column * 130, 195 + (row % 4) * 118)
            scene.append(make_preset_shape(name, center))
            preset_count += 1
            message = f"{PRESET_LABELS[name]} ajouté : pincez-le pour le déplacer"
        elif name == "finaliser":
            if not finalize_pending_shape(scene, pending_shape):
                message = "Dessinez une forme avant de l'ajouter"
            else:
                pending_shape = None
                message = "Dessin ajouté : pincez-le pour le déplacer"

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Impossible de lire l'image de la caméra.")
                frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                identify_faces(frame, recognizer, names, face_detector)
                draw_face_wireframe(frame, results)
                draw_pose_skeleton(frame, results.pose_landmarks)
                hand_gestures = [
                    gesture
                    for hand_name, hand_landmarks in (
                        ("Gauche", results.left_hand_landmarks),
                        ("Droite", results.right_hand_landmarks),
                    )
                    if (
                        gesture := extract_hand_gesture(
                            frame, hand_landmarks, hand_name
                        )
                    ) is not None
                ]
                pinching_hands = [
                    gesture for gesture in hand_gestures if gesture["pinching"]
                ]
                drawing_hands = [
                    gesture
                    for gesture in hand_gestures
                    if gesture["finger_states"].get("Index", False)
                    and not gesture["finger_states"].get("Majeur", False)
                ]
                active_hand = (
                    pinching_hands[0]
                    if pinching_hands
                    else drawing_hands[0]
                    if drawing_hands
                    else hand_gestures[0]
                    if hand_gestures
                    else None
                )
                tip = active_hand["tip"] if active_hand else None
                gesture_cursor = active_hand["cursor"] if active_hand else None
                finger_states = (
                    active_hand["finger_states"] if active_hand else {}
                )
                if tip is not None:
                    smoothed_tip = smooth_point(smoothed_tip, tip)
                    tip = smoothed_tip
                if gesture_cursor is not None:
                    smoothed_cursor = smooth_point(
                        smoothed_cursor, gesture_cursor
                    )
                    gesture_cursor = smoothed_cursor
                else:
                    smoothed_tip = None
                    smoothed_cursor = None
                pinching = bool(pinching_hands)

                selected = None
                if gesture_cursor is not None and pinching:
                    for name, (x1, y1, x2, y2) in buttons.items():
                        if (
                            x1 <= gesture_cursor[0] <= x2
                            and y1 <= gesture_cursor[1] <= y2
                        ):
                            selected = name
                            break
                if selected:
                    hold_count = hold_count + 1 if selected == held_button else 1
                    held_button = selected
                    if hold_count >= 12 and not button_fired:
                        activate_button(selected)
                        button_fired = True
                else:
                    held_button = None
                    hold_count = 0
                    button_fired = False

                if not pinching:
                    active_drag = None
                elif (
                    selected is None
                    and active_drag is None
                    and gesture_cursor is not None
                ):
                    active_drag = next(
                        (
                            index
                            for index in range(len(scene) - 1, -1, -1)
                            if hit_test_shape(scene[index], gesture_cursor)
                        ),
                        None,
                    )
                    if active_drag is not None:
                        drag_offset = (
                            np.asarray(scene[active_drag]["offset"])
                            - np.asarray(gesture_cursor)
                        )
                if (
                    pinching
                    and active_drag is not None
                    and gesture_cursor is not None
                ):
                    scene[active_drag]["offset"] = (
                        np.asarray(gesture_cursor) + drag_offset
                    ).tolist()

                interaction_point = gesture_cursor if pinching else tip
                over_button = interaction_point is not None and any(
                    x1 <= interaction_point[0] <= x2
                    and y1 <= interaction_point[1] <= y2
                    for x1, y1, x2, y2 in buttons.values()
                )
                can_draw = (
                    finger_states.get("Index", False)
                    and not finger_states.get("Majeur", False)
                    and not pinching
                    and not over_button
                    and tip is not None
                )
                if can_draw:
                    if not drawing:
                        stroke_points = [tip]
                    elif np.linalg.norm(np.asarray(tip) - stroke_points[-1]) >= 2:
                        stroke_points.append(tip)
                    drawing = True
                else:
                    if drawing and len(stroke_points) >= 2:
                        pending_shape = {
                            "points": [list(point) for point in stroke_points],
                            "offset": [0, 0],
                            "closed": True,
                        }
                        preview = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                        message = render_shape(preview, pending_shape)
                    stroke_points = []
                    drawing = False

                clicked_action = mouse_state["action"]
                mouse_state["action"] = None
                if clicked_action:
                    activate_button(clicked_action)
                completed_stroke = mouse_state["completed_stroke"]
                mouse_state["completed_stroke"] = None
                if completed_stroke and len(completed_stroke) >= 2:
                    pending_shape = {
                        "points": [list(point) for point in completed_stroke],
                        "offset": [0, 0],
                        "closed": True,
                    }
                    message = "Tracé prêt : cliquez sur Finaliser pour l'ajouter"

                canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                for shape in scene:
                    message = render_shape(canvas, shape)
                if pending_shape is not None:
                    message = render_shape(canvas, pending_shape)
                if drawing and len(stroke_points) >= 2:
                    preview_shape = {
                        "points": [list(point) for point in stroke_points],
                        "offset": [0, 0],
                        "closed": True,
                    }
                    message = render_shape(canvas, preview_shape)
                if mouse_state["mouse_drawing"] and len(mouse_state["mouse_points"]) >= 2:
                    mouse_preview = {
                        "points": [
                            list(point) for point in mouse_state["mouse_points"]
                        ],
                        "offset": [0, 0],
                        "closed": True,
                    }
                    message = render_shape(canvas, mouse_preview)

                output = cv2.addWeighted(frame, 0.65, canvas, 1.0, 0)
                for name, (x1, y1, x2, y2) in buttons.items():
                    if name == "effacer":
                        color = (70, 70, 180)
                    elif name == "finaliser":
                        color = (45, 95, 210)
                    else:
                        color = (90, 75, 45)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, -1)
                    label = button_labels[name]
                    cv2.putText(
                        output,
                        label,
                        (x1 + 5, y1 + 29),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (255, 255, 255),
                        2,
                    )
                if message:
                    cv2.putText(
                        output,
                        message,
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 255),
                        2,
                    )
                cv2.putText(
                    output,
                    "Souris : glisser et fermer | Main : index pointe, majeur plie | Finaliser : ajouter",
                    (20, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (255, 255, 255),
                    1,
                )
                if hand_gestures:
                    finger_status = " | ".join(
                        f"{gesture['name']}: "
                        + " ".join(
                            name
                            for name, extended
                            in gesture["finger_states"].items()
                            if extended
                        )
                        for gesture in hand_gestures
                    )
                    cv2.putText(
                        output,
                        f"Doigts tendus: {finger_status}",
                        (20, HEIGHT - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                body_status = (
                    f"Corps detecte | Mains: {len(hand_gestures)}/2"
                    if results.pose_landmarks is not None
                    else f"Corps non detecte | Mains: {len(hand_gestures)}/2"
                )
                cv2.putText(
                    output,
                    body_status,
                    (WIDTH - 180, HEIGHT - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
