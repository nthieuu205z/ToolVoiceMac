#!/bin/bash
# ============================================================
#  ToolVietSub — chạy bằng 1 cú nhấp chuột trên macOS
#  Bấm đúp file này để khởi động server và mở giao diện.
# ============================================================

set -u

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

PYTHON=".venv/bin/python"
PORT=8000

fail() {
  printf '\n[LỖI] %s\n\n' "$1"
  read -r -n 1 -s -p "Nhấn phím bất kỳ để đóng cửa sổ..."
  printf '\n'
  exit 1
}

if [ ! -x "$PYTHON" ]; then
  fail "Không tìm thấy Python trong .venv của project. Hãy cài đặt theo HUONG_DAN_MAC.md."
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  fail "Port $PORT đang được sử dụng. Hãy tắt server cũ trước khi chạy lại ToolVietSub."
fi

printf '%s\n' "============================================================"
printf '%s\n' "  ToolVietSub đang khởi động..."
printf '%s\n' "  Project: $PROJECT_DIR"
printf '%s\n' "  Giao diện: http://127.0.0.1:$PORT"
printf '%s\n' "  Nút Tắt tool sẽ dừng server và các process con của tool."
printf '%s\n\n' "============================================================"

# Chỉ mở trình duyệt sau khi server thực sự sẵn sàng.
# Tiến trình này là con của server sau lệnh exec, nên không bị bỏ lại khi shutdown.
(
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
      open "http://127.0.0.1:$PORT/" >/dev/null 2>&1
      exit 0
    fi
    sleep 1
  done
  printf '%s\n' "[LƯU Ý] Server chưa phản hồi sau 60 giây. Kiểm tra log trong cửa sổ này." >&2
) &

# exec để server thay thế shell launcher; nút shutdown sẽ nhận đúng PID gốc.
exec .venv/bin/python -m uvicorn backend.main:app --port 8000 --host 127.0.0.1
