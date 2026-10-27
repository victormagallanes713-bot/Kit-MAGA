#!/usr/bin/env bash
# Instala el CODIGO de LatentSync y su entorno. NO descarga pesos: reutiliza
# /workspace/models_repository/latentsync-1.5 (sin duplicar 9 GB).
set -uo pipefail
LS_DIR="${LATENTSYNC_DIR:-/workspace/LatentSync}"
VENV="${LATENTSYNC_VENV:-/workspace/venv_latentsync}"
PESOS="${LATENTSYNC_PESOS:-/workspace/models_repository/latentsync-1.5}"

echo "== pesos =="
[ -f "$PESOS/latentsync_unet.pt" ] || { echo "no encuentro $PESOS/latentsync_unet.pt. Contenido:"; find "$PESOS" -maxdepth 2 | head -20; exit 1; }
[ -f "$PESOS/whisper/tiny.pt" ] || echo "AVISO: falta $PESOS/whisper/tiny.pt (puede estar con otro nombre; revisa con find)"

echo "== codigo =="
if [ ! -d "$LS_DIR/.git" ]; then git clone --depth 1 https://github.com/bytedance/LatentSync "$LS_DIR" || exit 1; fi
if [ -d "$LS_DIR/checkpoints" ] && [ ! -L "$LS_DIR/checkpoints" ]; then rmdir "$LS_DIR/checkpoints" 2>/dev/null || mv "$LS_DIR/checkpoints" "$LS_DIR/checkpoints_vacio"; fi
ln -sfn "$PESOS" "$LS_DIR/checkpoints" && echo "checkpoints -> $PESOS"

echo "== entorno (Python 3.10 con uv; LatentSync fija mediapipe 0.10.11, que no existe para Python 3.12) =="
python3 --version
command -v uv >/dev/null || pip install -q uv || exit 1
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,10) else 1)"; then
  rm -rf "$VENV"; uv venv --python 3.10 "$VENV" || exit 1
fi
apt-get install -y -qq libgl1 >/dev/null 2>&1 || echo "AVISO: no se pudo instalar libgl1 (opencv puede fallar)"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
  -r "$LS_DIR/requirements.txt" || { echo "FALLO la instalacion; copia el error y enviamelo"; exit 1; }
"$VENV/bin/python" -c "import torch,mediapipe,decord,diffusers;print('torch',torch.__version__)" || exit 1

echo "== configuraciones disponibles =="
ls "$LS_DIR/configs/unet" 2>/dev/null || ls "$LS_DIR/configs"
echo "Listo. Prueba corta con GPU: bash run_latentsync.sh foto.png audio.wav salida.mp4"
