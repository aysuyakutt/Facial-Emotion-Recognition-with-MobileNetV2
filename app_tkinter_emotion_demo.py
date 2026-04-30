# app_emotion_tkinter.py
import os
import shutil
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------
# AYARLAR
# ----------------------------
IMG_SIZE = 224
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

UI_IMG_W = 380
UI_IMG_H = 380

# Modeli otomatik yüklemek için: script'in olduğu klasörde arar
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "final_emotion_model2.keras")


# ----------------------------
# 1) CUSTOM MODEL CLASS: AccumulationModel
# ----------------------------
class AccumulationModel(tf.keras.Model):
    def __init__(self, accum_steps=2, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps = accum_steps

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"accum_steps": self.accum_steps})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ----------------------------
# 2) PATCH: DepthwiseConv2D groups fix
# ----------------------------
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D


class DepthwiseConv2DPatched(_DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return cls(**config)


CUSTOM_OBJECTS = {
    "AccumulationModel": AccumulationModel,
    "DepthwiseConv2D": DepthwiseConv2DPatched,
    "DepthwiseConv2DPatched": DepthwiseConv2DPatched,
}


# ----------------------------
# TFLite Runner
# ----------------------------
class TFLiteRunner:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.in_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        self.input_index = self.in_details[0]["index"]
        self.output_index = self.out_details[0]["index"]
        self.input_dtype = self.in_details[0]["dtype"]

    def predict(self, x_float32: np.ndarray, raw_rgb_uint8: np.ndarray):
        # Quantized model ise uint8 ister
        if self.input_dtype == np.uint8:
            x_in = raw_rgb_uint8.astype(np.uint8)
        else:
            x_in = x_float32.astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_index, x_in)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out


# ----------------------------
# MODEL LOADER
# ----------------------------
def _read_magic4(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read(4)


def load_model_any(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")

    # SavedModel klasörü seçildiyse (opsiyonel)
    if os.path.isdir(path):
        loaded = tf.saved_model.load(path)
        sigs = list(loaded.signatures.keys())
        if not sigs:
            raise RuntimeError("SavedModel signatures boş. Klasör bozuk olabilir.")
        endpoint = "serving_default" if "serving_default" in sigs else sigs[0]

        layer = tf.keras.layers.TFSMLayer(path, call_endpoint=endpoint)
        inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        out = layer(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        return tf.keras.Model(inp, out)

    lower = path.lower()

    # TFLite
    if lower.endswith(".tflite"):
        return TFLiteRunner(path)

    # magic bytes kontrol
    head = _read_magic4(path)

    # HDF5 (b'\x89HDF') ise
    if head == b"\x89HDF":
        # Uzantı yanlışsa (.keras yazıp HDF5 kaydetmiş olabilirsin)
        if not (lower.endswith(".h5") or lower.endswith(".hdf5")):
            tmp_h5 = os.path.join(tempfile.gettempdir(), "temp_model_fix.h5")
            shutil.copyfile(path, tmp_h5)
            return tf.keras.models.load_model(
                tmp_h5, compile=False, custom_objects=CUSTOM_OBJECTS
            )
        return tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)

    # Diğer durumlar (gerçek .keras zip vb.)
    return tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)


# ----------------------------
# IMAGE PREP
# ----------------------------
def prepare_inputs(pil_img: Image.Image):
    """
    2 input döner:
      - x_float32: preprocess_input uygulanmış (1,224,224,3) float32
      - raw_uint8: (1,224,224,3) uint8 [0..255]  (quant tflite için)
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    raw = np.array(img, dtype=np.uint8)
    raw = np.expand_dims(raw, axis=0)

    x = raw.astype(np.float32)
    x = preprocess_input(x)
    return x, raw


def softmax_if_needed(scores: np.ndarray) -> np.ndarray:
    scores = np.array(scores).astype(np.float32)
    if scores.ndim == 2:
        scores = scores[0]
    elif scores.ndim > 2:
        scores = scores.reshape(-1)

    s = float(scores.sum())
    if scores.min() >= 0.0 and scores.max() <= 1.0 and abs(s - 1.0) < 1e-2:
        return scores

    ex = np.exp(scores - np.max(scores))
    return ex / (ex.sum() + 1e-8)


# ----------------------------
# CAMERA HELPERS (cv2)
# ----------------------------
def _largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return faces[0]


def _crop_with_margin_rgb(rgb: np.ndarray, box, margin=0.25):
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)

    crop = rgb[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)


def _blur_score_laplacian(cv2, rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _enhance_lighting_clahe(cv2, rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out


# ----------------------------
# TKINTER UI
# ----------------------------
class EmotionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Recognition (MobileNetV2)")
        self.geometry("960x580")
        self.resizable(False, False)

        self.model_path = DEFAULT_MODEL_PATH
        self.model_obj = None  # tf.keras.Model veya TFLiteRunner

        self.current_pil = None
        self.current_tk = None

        # Camera state
        self._cv2 = None
        self._face_cascade = None

        self._cam_win = None
        self._cam_label = None
        self._cam_tk = None
        self._cam_cap = None
        self._cam_running = False
        self._cam_paused = False  # ✅ predict sırasında loop çatışmasın
        self._cam_last_rgb = None
        self._cam_last_face_box = None

        # Kamera butonları (disable/enable için)
        self._btn_cam_capture = None
        self._btn_cam_capture_predict = None
        self._btn_cam_close = None
        self._cam_busy = False  # ✅ art arda tıklamalarda re-entrancy olmasın

        # Kamera ayarları
        self.cam_use_face_crop = tk.BooleanVar(value=True)
        self.cam_use_clahe = tk.BooleanVar(value=True)
        self.cam_draw_bbox = tk.BooleanVar(value=True)
        self.cam_margin = tk.DoubleVar(value=0.25)

        # Performans: her N framede bir face detect
        self._detect_every_n = 3
        self._frame_count = 0

        self._build_ui()

        # ✅ Açılışta otomatik yükle
        if self.model_path and os.path.exists(self.model_path):
            self._try_load(self.model_path)
        else:
            self.status.config(text="Model bulunamadı. 'Model Dosyası Seç' ile seç.")

    def _build_ui(self):
        top = tk.Frame(self, padx=12, pady=10)
        top.pack(fill="x")

        self.model_label = tk.Label(top, text="Model: (seçilmedi)", anchor="w")
        self.model_label.pack(side="left", fill="x", expand=True)

        tk.Button(
            top,
            text="Model Dosyası Seç (.h5/.keras/.tflite)",
            command=self.select_model_file,
        ).pack(side="right")

        mid = tk.Frame(self, padx=12, pady=8)
        mid.pack(fill="both", expand=True)

        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=False)

        self.img_panel = tk.Label(left, text="Görsel seçilmedi", bd=1, relief="solid")
        self.img_panel.pack()

        btns = tk.Frame(left, pady=10)
        btns.pack(fill="x")

        tk.Button(btns, text="Görsel Seç", command=self.select_image).pack(side="left")
        tk.Button(btns, text="Kamera", command=self.open_camera).pack(side="left", padx=8)
        tk.Button(btns, text="Tahmin Et", command=self.predict).pack(side="left", padx=8)
        tk.Button(btns, text="Temizle", command=self.clear).pack(side="left")

        right = tk.Frame(mid, padx=14)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="Sonuç", font=("Arial", 16, "bold")).pack(anchor="w", pady=(0, 8))
        self.pred_label = tk.Label(right, text="—", font=("Arial", 22, "bold"))
        self.pred_label.pack(anchor="w", pady=(0, 12))

        tk.Label(right, text="Sınıf Olasılıkları:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.prob_text = tk.Text(right, height=16, width=52)
        self.prob_text.pack(fill="both", expand=True, pady=(6, 0))
        self.prob_text.configure(state="disabled")

        bottom = tk.Frame(self, padx=12, pady=8)
        bottom.pack(fill="x")
        self.status = tk.Label(bottom, text="Hazır.", anchor="w")
        self.status.pack(fill="x")

    def _try_load(self, path: str):
        try:
            self.status.config(text="Model yükleniyor...")
            self.update_idletasks()

            obj = load_model_any(path)
            self.model_obj = obj
            self.model_path = path

            self.model_label.config(text=f"Model: {self.model_path}")
            self.status.config(text="Model yüklendi ✅")
        except Exception as e:
            self.model_obj = None
            self.status.config(text="Model yüklenemedi ❌")
            messagebox.showerror(
                "Model Hatası", f"{e}\n\n📌 ÖNERİ: En stabil çözüm: .tflite modeli seç."
            )

    def select_model_file(self):
        path = filedialog.askopenfilename(
            title="Model seç",
            filetypes=[("Model", "*.tflite *.h5 *.hdf5 *.keras"), ("All files", "*.*")],
        )
        if not path:
            return
        self._try_load(path)

    def _set_ui_image_fixed(self, pil_img: Image.Image):
        show = pil_img.convert("RGB").resize((UI_IMG_W, UI_IMG_H), Image.BILINEAR)
        self.current_tk = ImageTk.PhotoImage(show)
        self.img_panel.configure(image=self.current_tk, text="")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Görsel seç",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pil = Image.open(path)
            self.current_pil = pil
            self._set_ui_image_fixed(pil)
            self.status.config(text=f"Görsel seçildi: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Hata", f"Görsel açılamadı:\n{e}")

    # ----------------------------
    # CAMERA
    # ----------------------------
    def _ensure_cv2(self):
        if self._cv2 is not None:
            return True
        try:
            import cv2  # noqa
            self._cv2 = cv2
            return True
        except Exception:
            messagebox.showerror(
                "OpenCV yok",
                "Kamera + yüz tespiti için OpenCV (cv2) gerekli.\n\nKurulum:\n  pip install opencv-python\n",
            )
            return False

    def _init_face_detector(self):
        if self._cv2 is None:
            return
        try:
            cascade_path = os.path.join(
                self._cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self._face_cascade = self._cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                self._face_cascade = None
        except Exception:
            self._face_cascade = None

    def _set_cam_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in (self._btn_cam_capture, self._btn_cam_capture_predict, self._btn_cam_close):
            if b is not None:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

    def open_camera(self):
        if not self._ensure_cv2():
            return

        if self._cam_win is not None and tk.Toplevel.winfo_exists(self._cam_win):
            self._cam_win.lift()
            return

        self._init_face_detector()

        cap = self._cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Kamera Hatası", "Kamera açılamadı (VideoCapture(0) başarısız).")
            return

        try:
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass

        self._cam_cap = cap
        self._cam_running = True
        self._cam_paused = False
        self._cam_last_rgb = None
        self._cam_last_face_box = None
        self._frame_count = 0
        self._cam_busy = False

        win = tk.Toplevel(self)
        win.title("Kamera")
        win.geometry("700x720")
        win.resizable(False, False)
        self._cam_win = win

        self._cam_label = tk.Label(win, text="Kamera görüntüsü yükleniyor...", bd=1, relief="solid")
        self._cam_label.pack(padx=10, pady=10)

        opts = tk.LabelFrame(win, text="Kamera İyileştirmeleri", padx=8, pady=6)
        opts.pack(fill="x", padx=10, pady=(0, 8))

        tk.Checkbutton(opts, text="Yüzü otomatik crop'la (en etkili)", variable=self.cam_use_face_crop).pack(anchor="w")
        tk.Checkbutton(opts, text="Işık/kontrast düzelt (CLAHE)", variable=self.cam_use_clahe).pack(anchor="w")
        tk.Checkbutton(opts, text="Yüz kutusunu göster", variable=self.cam_draw_bbox).pack(anchor="w")

        row = tk.Frame(opts)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text="Crop margin:").pack(side="left")
        tk.Scale(
            row, from_=0.0, to=0.6, resolution=0.05, orient="horizontal",
            variable=self.cam_margin, length=220
        ).pack(side="left", padx=8)
        tk.Label(row, text="(saç/çene kalsın)").pack(side="left")

        controls = tk.Frame(win)
        controls.pack(fill="x", padx=10, pady=8)

        self._btn_cam_capture = tk.Button(
            controls, text="Foto Çek",
            command=lambda: self.capture_from_camera(predict_after=False)
        )
        self._btn_cam_capture.pack(side="left")

        self._btn_cam_capture_predict = tk.Button(
            controls, text="Çek + Tahmin (kapat)",
            command=lambda: self.capture_from_camera(predict_after=True)
        )
        self._btn_cam_capture_predict.pack(side="left", padx=8)

        self._btn_cam_close = tk.Button(controls, text="Kapat", command=self.close_camera)
        self._btn_cam_close.pack(side="right")

        hint = tk.Label(
            win,
            text="İpucu: En iyi sonuç için yüz kadrajda büyük + önden ışık + sabit dur.",
            anchor="w",
        )
        hint.pack(fill="x", padx=10)

        win.protocol("WM_DELETE_WINDOW", self.close_camera)

        self.status.config(text="Kamera açıldı ✅")
        self._camera_loop()

    def _detect_face_box(self, frame_rgb: np.ndarray):
        if self._face_cascade is None:
            return None
        gray = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return _largest_face(faces)

    def _grab_frame_rgb_now(self):
        """✅ capture anında kameradan fresh frame al (bugları azaltır)."""
        if self._cam_cap is None:
            return None
        ok, frame_bgr = self._cam_cap.read()
        if not ok:
            return None
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def _camera_loop(self):
        if not self._cam_running or self._cam_cap is None or self._cam_win is None:
            return

        # ✅ Predict/capture sırasında loop duraklat (race/çakışma azalır)
        if self._cam_paused:
            if self._cam_win is not None:
                self._cam_win.after(30, self._camera_loop)
            return

        ok, frame_bgr = self._cam_cap.read()
        if not ok:
            if self._cam_win is not None:
                self._cam_win.after(30, self._camera_loop)
            return

        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        self._cam_last_rgb = frame_rgb

        self._frame_count += 1
        if self.cam_use_face_crop.get() and (self._frame_count % self._detect_every_n == 0):
            self._cam_last_face_box = self._detect_face_box(frame_rgb)

        preview = frame_rgb.copy()
        if self.cam_draw_bbox.get() and self._cam_last_face_box is not None:
            x, y, w, h = self._cam_last_face_box
            self._cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        preview_w, preview_h = 600, 400
        pil = Image.fromarray(preview).resize((preview_w, preview_h), Image.BILINEAR)
        self._cam_tk = ImageTk.PhotoImage(pil)
        self._cam_label.configure(image=self._cam_tk, text="")

        if self._cam_win is not None:
            self._cam_win.after(30, self._camera_loop)

    def _make_capture_pil(self, frame_rgb: np.ndarray) -> Image.Image:
        rgb = frame_rgb

        # Capture anında da face box güncelle (Foto Çek -> Çek+Tahmin sırasındaki hataları azaltır)
        if self.cam_use_face_crop.get():
            try:
                box_now = self._detect_face_box(rgb)
                if box_now is not None:
                    self._cam_last_face_box = box_now
            except Exception:
                pass

        # Blur ölç
        try:
            bs = _blur_score_laplacian(self._cv2, rgb)
        except Exception:
            bs = None

        # Yüz crop
        if self.cam_use_face_crop.get() and self._cam_last_face_box is not None:
            try:
                crop_rgb, _ = _crop_with_margin_rgb(
                    rgb, self._cam_last_face_box, margin=float(self.cam_margin.get())
                )
                if crop_rgb.shape[0] >= 60 and crop_rgb.shape[1] >= 60:
                    rgb = crop_rgb
            except Exception:
                pass

        # Işık düzelt
        if self.cam_use_clahe.get():
            try:
                rgb = _enhance_lighting_clahe(self._cv2, rgb)
            except Exception:
                pass

        # Status mesajı
        if bs is not None and bs < 60:
            self.status.config(text=f"Uyarı: görüntü bulanık olabilir (blur={bs:.0f}). Sabit dur/ışık artır.")
        else:
            self.status.config(text="Kameradan foto çekildi ✅")

        return Image.fromarray(rgb)

    def capture_from_camera(self, predict_after: bool = False):
        # ✅ aynı anda iki kere tıklama / üst üste çağrı engeli
        if self._cam_busy:
            return
        self._cam_busy = True
        self._set_cam_buttons_state(False)
        self._cam_paused = True  # loop duraklat

        try:
            # ✅ öncelik: fresh frame
            frame_rgb = self._grab_frame_rgb_now()
            if frame_rgb is None:
                frame_rgb = self._cam_last_rgb

            if frame_rgb is None:
                messagebox.showwarning("Uyarı", "Henüz kamera görüntüsü alınamadı.")
                return

            pil = self._make_capture_pil(frame_rgb)

            self.current_pil = pil
            self._set_ui_image_fixed(pil)

            if predict_after:
                # ✅ Çek + Tahmin + Kapat
                self.predict()
                self.close_camera(silent=True)  # status'ı ezmesin

        finally:
            # Kamera kapanmış olabilir; kapanmadıysa devam ettir
            if self._cam_running and self._cam_win is not None:
                self._cam_paused = False
                self._set_cam_buttons_state(True)
            self._cam_busy = False

    def close_camera(self, silent: bool = False):
        self._cam_running = False
        self._cam_paused = False

        if self._cam_cap is not None:
            try:
                self._cam_cap.release()
            except Exception:
                pass
            self._cam_cap = None

        if self._cam_win is not None:
            try:
                self._cam_win.destroy()
            except Exception:
                pass
            self._cam_win = None

        self._cam_label = None
        self._cam_tk = None
        self._cam_last_rgb = None
        self._cam_last_face_box = None

        self._btn_cam_capture = None
        self._btn_cam_capture_predict = None
        self._btn_cam_close = None
        self._cam_busy = False

        if not silent:
            self.status.config(text="Kamera kapatıldı.")

    # ----------------------------
    # PREDICT
    # ----------------------------
    def predict(self):
        if self.model_obj is None:
            messagebox.showerror("Hata", "Model yüklü değil. Önce model seç.")
            return
        if self.current_pil is None:
            messagebox.showerror("Hata", "Önce bir görsel seç (veya kamera ile çek).")
            return

        try:
            self.img_panel.configure(image=self.current_tk)

            x_float32, raw_uint8 = prepare_inputs(self.current_pil)

            if isinstance(self.model_obj, TFLiteRunner):
                out = self.model_obj.predict(x_float32, raw_uint8)
            else:
                out = self.model_obj.predict(x_float32, verbose=0)

            probs = softmax_if_needed(out)

            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
            pred_conf = float(probs[pred_idx])

            self.pred_label.config(text=f"{pred_name}  ({pred_conf:.3f})")

            pairs = []
            for i, p in enumerate(probs):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
                pairs.append((name, float(p)))
            pairs.sort(key=lambda t: t[1], reverse=True)

            self.prob_text.configure(state="normal")
            self.prob_text.delete("1.0", tk.END)
            for name, p in pairs:
                self.prob_text.insert(tk.END, f"{name:<10} : {p:.4f}\n")
            self.prob_text.configure(state="disabled")

            self.status.config(text="Tahmin tamam ✅")

        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin hatası:\n{e}")
            self.status.config(text="Tahmin hatası ❌")

    def clear(self):
        self.current_pil = None
        self.current_tk = None
        self.img_panel.configure(image="", text="Görsel seçilmedi")

        self.pred_label.config(text="—")
        self.prob_text.configure(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.configure(state="disabled")

        self.status.config(text="Temizlendi.")


if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
