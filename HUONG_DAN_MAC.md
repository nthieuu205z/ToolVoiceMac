# Cài ToolVietSub trên macOS — Apple Silicon (M1/M2/M3/M4)

Hướng dẫn này dành cho MacBook/Mac mini chạy chip **Apple Silicon**. Chip M-series có
**Metal (MPS)** có thể tăng tốc **bước tạo giọng đọc OmniVoice** — chạy bằng PyTorch trên GPU tích hợp.
Ứng dụng nạp model trên CPU rồi chuyển sang Metal để tránh đường dispatch trực tiếp đã từng
crash native. Không coi phép probe MPS nhỏ là bằng chứng toàn bộ model ổn định;
Whisper cũng chạy CPU. Nếu OmniVoice đang được chọn, lần đầu nạp model có thể mất vài giây
trước khi xuất audio; terminal sẽ ghi rõ mốc nạp và suy luận. Bước **nhận diện giọng nói (Whisper)**
chạy trên CPU (bản CTranslate2 cho Mac chưa có backend Metal/CoreML) — vẫn nhanh trên
chip M-series nhờ int8, chỉ bước dịch cần mạng (Gemini).

## Cần chuẩn bị

- macOS 13+ trên Apple Silicon (Intel cũng chạy được nhưng không có Metal — TTS lùi về
  CPU, chậm hơn nhiều).
- Khoảng 5 GB đĩa trống cho bộ cơ bản; cần thêm khoảng 3,3 GB nếu dùng OmniVoice.
- Homebrew (nếu chưa có: <https://brew.sh>).
- Khóa Gemini API (dùng lại khóa trong `.env` từ máy cũ).

## 1. Chép dự án từ máy cũ

Copy cả thư mục `ToolVietSub` sang máy mới, **trừ** các thư mục sau (đừng chép):

| Chép | KHÔNG chép |
|---|---|
| toàn bộ code | `.venv/` — binary của hệ điều hành khác, vô dụng trên macOS |
| `.env` — chứa khóa API, giữ kín, đừng gửi qua kênh công khai | `jobs/` — kết quả cũ, chép hay không tùy bạn |
| `custom_voices/` — giọng nhân bản của bạn nằm ở đây, sang máy mới dùng tiếp được | `__pycache__/`, `.pytest_cache/`, `graphify-out/` |
| `web/static/previews/` — file nghe thử của giọng (tùy chọn) | cache Hugging Face (`~/.cache/huggingface`) — tải lại sạch hơn |

## 2. Cài công cụ nền

Mở **Terminal** và chạy:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # chỉ lần đầu
brew install python@3.12 ffmpeg uv
```

Kiểm tra đủ ba thứ:

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 --version   # Python 3.12.x
ffmpeg -version                                          # ffmpeg version ...
uv --version                                             # uv x.x.x
```

Nếu `ffmpeg` không được nhận: mở `.env`, điền đường dẫn đầy đủ vào `FFMPEG_BIN` và
`FFPROBE_BIN` (thường là `/opt/homebrew/bin/ffmpeg` và `/opt/homebrew/bin/ffprobe`).

## 3. Môi trường Python

```bash
cd ~/duong/dan/toi/ToolVietSubMac
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv    # hoặc: uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,omnivoice]"
```

Lệnh trên cài OmniVoice cho giọng nhân bản; edge-tts vẫn cần mạng khi tạo giọng đọc.

## 4. Bật Metal (MPS)

Trên macOS, bản PyTorch mặc định từ PyPI **đã hỗ trợ Metal** — khác Windows, không cần
thay index. Extra `omnivoice` đã cài các thư viện cần thiết. Ứng dụng thử một phép tính nhỏ
trên MPS lúc khởi động; phép thử chỉ xác nhận kernel cơ bản, không bảo đảm mọi kernel của
OmniVoice tương thích:

Kiểm tra — dòng dưới nên in `True`:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 4b. (Tùy chọn) OmniVoice — giọng nhân bản chất lượng cao

Mặc định `CLONE_TTS_PROVIDER=omnivoice`: khi bạn chọn một **giọng nhân bản**, tool đọc
bằng OmniVoice (nhân bản zero-shot); giọng **dựng sẵn** vẫn dùng `TTS_PROVIDER`.
Nếu chưa cài OmniVoice, tính năng nhân bản sẽ được tắt và API báo trạng thái rõ ràng.

Cài (chỉ phần *inference*, né xung đột `numpy 2.x` ↔ `librosa/numba` mà ta không cần). Bước 3
đã cài torch và transformers; OmniVoice cần thêm accelerate:

```bash
uv pip install omnivoice --no-deps
uv pip install accelerate
```

Model **~3,3 GB** tự tải lần đầu, hoặc bấm nút tải trong mục **"Model trên máy"**
(có thanh %). OmniVoice chạy fp16 trên Metal khi smoke test xác nhận ổn định; với MPS, model được nạp trên
CPU rồi chuyển sang Metal để tránh lỗi dispatch native đã tái hiện. Smoke test hiện tại đã
chạy batch hai câu thành công trong process sạch; vẫn nên kiểm tra lại sau khi đổi PyTorch.
Lời của clip mẫu (`ref_text`) do Whisper chép một lần rồi nhớ cạnh clip.

> **License:** trọng số OmniVoice là **CC-BY-NC (phi thương mại)** — dùng cá nhân thoải
> mái, nhưng nếu thương mại hoá thì đây là ràng buộc.

Đặt `CLONE_TTS_PROVIDER=none` để tắt nhân bản trong `.env`.

## 5. Kiểm tra `.env`

Nếu đã chép `.env` từ máy cũ thì giữ nguyên. Nếu làm mới: `cp .env.example .env`
rồi điền `GEMINI_API_KEY`. Cấu hình khuyến nghị (đã là mặc định trong `.env` hiện tại):

```
GEMINI_BACKEND=vertex        # khóa tạo trong Google Cloud; khóa AI Studio thì để developer
STT_PROVIDER=whisper         # nhận diện trên CPU int8 (miễn phí) + cho mốc từng từ để cắt câu
STT_WORKERS=1                # Whisper tự tuần tự hóa bên trong; để 1 cho khỏi tranh CPU
TTS_PROVIDER=edge            # giọng dựng sẵn; clone dùng OmniVoice
SENTENCE_LEVEL_TIMING=true   # đọc theo từng câu theo mốc từng từ → bám hình sát
MAX_UTTERANCE_SECONDS=12     # trần vùng ffmpeg TRƯỚC khi tách câu, đừng tăng nếu không có lý do
```

## 6. Chạy

Cách 1 — **nhấp đúp** file `ChayTool.command` trong thư mục dự án. Lần đầu macOS hỏi
bảo mật: **nhấp chuột phải** vào file → **Mở** → xác nhận một lần.

Cách 2 — Terminal:

```bash
source .venv/bin/activate
uvicorn backend.main:app --port 8000
```

Mở <http://localhost:8000>. Server chỉ nghe trên máy của bạn (localhost), không mở ra mạng.

## 7. Lần chạy đầu tiên

1. Vào mục **"Model trên máy"** trên giao diện, bấm tải model OmniVoice (~3,3 GB), đợi 100%.
2. Chạy thử một video **ngắn** trước. Trong tab **Video Dubbing**, chọn ngôn ngữ đích
   `vi-VN` hoặc `en-US`. Xác nhận terminal ghi `Đang nạp OmniVoice trên mps`
   và job tạo đủ audio; nếu process thoát với `SIGSEGV (-11)`, báo lại log để điều tra
   phiên bản PyTorch/model trước khi chạy video dài.
3. Muốn xem GPU bận không: mở **Activity Monitor** → tab **Energy** (hoặc **GPU History**)
   trong bước "Đang tạo giọng đọc".

## 8. Cách dùng hằng ngày

Giao diện có hai workflow trong **New Job**:

- **Video Dubbing**: kéo thả (.mp4, .mkv, .mov… tối đa 8 GB), chọn `vi-VN` hoặc `en-US`, có phần trăm tải lên. Workflow này cần Gemini để dịch sang ngôn ngữ đích.
- **Text → Voice**: nhập tối đa **50.000 ký tự**, chọn ngôn ngữ/giọng, nghe giọng mẫu hoặc nghe thử nội dung, rồi nhận cả **WAV** và **MP3**. Edge hoặc OmniVoice local có thể chạy workflow này mà không cần Gemini key.
- **Chọn giọng**: các giọng Edge/Gemini dựng sẵn và giọng clone OmniVoice.
- **Voice Lab**: chọn ngôn ngữ nghe thử; trạng thái demo hiển thị đang tạo/sẵn sàng/tạo lại.
- **Nhân bản giọng**: bấm *＋ Nhân bản giọng*, đặt tên + tải mẫu audio **3–8 giây, một
  người nói, ít tạp âm**. Chỉ dùng giọng bạn có quyền sử dụng.
- **Bắt đầu xử lý**: cả Video Dubbing và Text → Voice vào cùng hàng đợi. Graph hiển thị 7 bước video hoặc 4 bước text. Chạy song song tối đa 2 job,
  job nộp thêm tự xếp hàng; bấm **Hủy** dừng được giữa chừng; đóng trình duyệt hay
  khởi động lại máy vẫn còn nguyên danh sách.
- **Xong**: video job cho tải video + `.srt`; text job cho tải WAV + MP3. Nếu có đoạn bị thiếu (mạng chập chờn…)
  màn kết quả sẽ báo đỏ kèm mốc thời gian — chạy lại video thường khắc phục được.
- Đổi cấu hình: sửa `.env` → `Ctrl+C` tắt server → chạy lại.

Muốn thêm ngôn ngữ mới, phải cập nhật registry `pipeline/languages.py`, capability của giọng
và provider, preview text và tests; không chỉ thêm một option HTML.

## 9. Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| macOS chặn `ChayTool.command` ("unidentified developer") | Nhấp **chuột phải** → **Mở** → xác nhận một lần; hoặc System Settings → Privacy & Security → Mở Anyway |
| `torch.backends.mps.is_available()` in `False` | torch đang là bản CPU hoặc sai kiến trúc — `uv pip uninstall -y torch && uv pip install torch` (PyPI macOS ARM thường có MPS) |
| Thiếu thư viện OmniVoice | `uv pip install -e ".[dev,omnivoice]"` rồi khởi động lại server |
| Báo lỗi kernel MPS lạ lẫm khi chạy TTS | Ứng dụng thử MPS lúc khởi động và sẽ lùi toàn bộ TTS về CPU nếu phép thử không chạy; nếu lỗi chỉ xuất hiện trong model, nâng macOS/PyTorch lên bản mới nhất |
| Lần đầu nạp model rất chậm | Kiểm tra mạng; cache nằm ở `~/.cache/huggingface`, lần sau không tải lại |
| Cổng 8000 bận | `uvicorn backend.main:app --port 8080`, hoặc tắt cửa sổ server cũ |
| OmniVoice crash trên MPS | Chạy lại smoke test trong process sạch; nếu vẫn `SIGSEGV (-11)`, dùng CPU tạm thời và ghi lại phiên bản PyTorch/macOS |
