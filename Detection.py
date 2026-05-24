import cv2
import numpy as np
from collections import deque
from inference_sdk import InferenceHTTPClient

# ── Configurações ──────────────────────────────────────────────
API_KEY       = "Wz9u3mWCzqhSMIQ6nbvp"
MODEL_ID      = "formas-eh9u9/1"
CONFIANCA_MIN = 0.5
ESTABILIDADE  = 3  # frames para confirmar detecção
FRAMES_LIMPAR = 4  # frames sem detecção para limpar a tela

# ── Cores por classe (BGR) ─────────────────────────────────────
CORES = {
    "Circulo":   (0, 165, 255),
    "Quadrado":  (0, 255,   0),
    "Triangulo": (255,  0,  0),
}

print("Conectando ao modelo...")
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)
print("Pronto! Pressione Q para sair.")

cap           = cv2.VideoCapture(0)
contador      = 0
deteccoes_est = []
historico     = deque(maxlen=ESTABILIDADE)
frames_vazio  = 0  # conta quantos frames seguidos vieram sem detecção

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    contador += 1

    if contador % 5 == 0:
        cv2.imwrite("temp_frame.jpg", frame)
        resultado = client.infer("temp_frame.jpg", model_id=MODEL_ID)

        deteccoes_raw = []
        for pred in resultado.get("predictions", []):
            if pred["confidence"] < CONFIANCA_MIN:
                continue
            cx = pred["x"]; cy = pred["y"]
            bw = pred["width"]; bh = pred["height"]
            x1 = int(cx - bw/2); y1 = int(cy - bh/2)
            x2 = int(cx + bw/2); y2 = int(cy + bh/2)
            dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
            deteccoes_raw.append({
                "bbox": (x1, y1, x2, y2),
                "classe": pred["class"],
                "conf": pred["confidence"],
                "dist": dist,
            })

        deteccoes_raw.sort(key=lambda d: d["dist"])
        classes_agora = tuple(d["classe"] for d in deteccoes_raw)
        historico.append(classes_agora)

        if len(deteccoes_raw) == 0:
            frames_vazio += 1
            # Limpa a tela após N frames sem detecção
            if frames_vazio >= FRAMES_LIMPAR:
                deteccoes_est = []
                historico.clear()
        else:
            frames_vazio = 0
            # Só atualiza se detecção for estável
            if len(historico) == ESTABILIDADE and len(set(historico)) == 1:
                deteccoes_est = deteccoes_raw

    # ── Desenha as caixas estabilizadas ───────────────────────
    for i, d in enumerate(deteccoes_est):
        x1, y1, x2, y2 = d["bbox"]
        cor       = CORES.get(d["classe"], (200, 200, 200))
        espessura = 3 if i == 0 else 1
        prefixo   = "[PROXIMO] " if i == 0 else ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, espessura)
        cv2.putText(frame, f"{prefixo}{d['classe']} {d['conf']:.0%}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    cv2.putText(frame, f"Objetos: {len(deteccoes_est)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if deteccoes_est:
        melhor = deteccoes_est[0]
        cv2.putText(frame, f"Pegar: {melhor['classe']} ({melhor['conf']:.0%})",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Deteccao de Formas", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
