# ToolVoiceMac

Voice cloning, Text → Voice và Video Dubbing cho **macOS trên Apple Silicon**. Cài bằng Terminal, sau đó gõ `toolvoice` để mở giao diện web trên máy.

## Yêu cầu

- Mac Apple Silicon (M1 trở lên), macOS 14+.
- Python 3.12; installer dùng Python 3.12.14 qua uv.
- FFmpeg/ffprobe. Nếu chưa có, installer cài bằng Homebrew khi `brew` đã được cài.
- Internet để cài thư viện và tải model lần đầu. Runtime hiệu năng được kiểm chứng trên M1 Max 64 GB; tốc độ và mức dùng bộ nhớ thay đổi theo máy/nội dung.

## Cài và chạy

Sau khi tag `v0.2.0` được phát hành và repository được public:

```bash
curl -fsSL https://raw.githubusercontent.com/nthieuu205z/ToolVoiceMac/v0.2.0/install.sh | bash
```

Sau khi cài:

```bash
toolvoice
```

Tool khởi động server tại `http://127.0.0.1:8000` và mở trình duyệt. Gọi lại lệnh sẽ mở instance đang chạy cùng thư mục dữ liệu. Nhấn `Ctrl+C` trong Terminal hoặc dùng nút tắt tool để dừng.

```bash
toolvoice --doctor                 # kiểm tra môi trường, không nạp model
toolvoice --no-browser             # chỉ chạy server
toolvoice --port 8001              # dùng cổng khác
toolvoice --data-dir '/path/data'  # kho dữ liệu riêng
```

Trong giai đoạn repository còn private, tài khoản có quyền truy cập có thể cài từ checkout:

```bash
uv tool install --python 3.12.14 --with-requirements requirements-macos.lock .
uv tool update-shell
```

Mở Terminal mới nếu lệnh chưa có trong PATH. Installer cho mỗi bản phát hành giữ phiên bản thư viện theo lock; cập nhật bằng installer của tag mới đã kiểm chứng.

## Sử dụng

- **Voice Lab:** tải giọng mẫu, đặt tên, chọn ngôn ngữ đầu ra Việt (`vi-VN`)/Anh (`en-US`) và nhãn. Chỉ sử dụng mẫu giọng bạn có quyền sử dụng.
- **Text → Voice:** đặt tên job, chọn giọng/ngôn ngữ, nhập tối đa 50.000 ký tự cho một job đơn và xuất WAV/MP3.
- **Project dài:** tối đa **200.000 ký tự**; văn bản trên 50.000 ký tự được chia thành các phần tối đa 5.000 ký tự, xử lý và xuất theo thứ tự. Tải project thành ZIP gồm thư mục audio được đánh số. Đây là giới hạn nội dung project, không phải model nhận một context 200k trong một lần.
- **Job Queue:** job mới ở trên cùng; có tiến độ, hủy và lịch sử.
- **Workspace:** cấu hình runtime, quản lý model và Gemini API key. Key cần cho dịch video hoặc provider Gemini; TTS clone local không cần key Gemini.
- **Video Dubbing:** nhận diện Whisper, dịch Gemini, tổng hợp giọng, ghép video và phụ đề.

Tải OmniVoice và Whisper trong mục model nếu chưa có. Model có sẵn được nạp nền khi khởi động. Cấu hình giọng và nội dung quyết định chất lượng; tool không huấn luyện lại trọng số khi clone.

## Runtime đã chốt

- OmniVoice 0.2.1, torch 2.13.0, torchaudio 2.11.0, transformers 5.16.1.
- Model MPS/FP16 và codec MPS; `split-cfg-rms-gqa-rope`, batch tối đa 2.
- 32 bước, guidance 2.0, speed 1.0; giữ cách chia câu và khoảng nghỉ.
- Whisper small/int8 trên CPU, gom feature chunks tối đa 8.
- Không bật thử nghiệm gộp câu, SiLU hoặc gộp Q/K normalization + rotary.

Chi tiết: [runtime](docs/RUNTIME.md), [cài đặt trên Mac](HUONG_DAN_MAC.md), [baseline phiên bản](docs/release/CURRENT_RUNTIME.json).

## Dữ liệu và kết nối mạng

Dữ liệu mặc định nằm trong `~/Library/Application Support/ToolVoiceMac/`: `.env`, `jobs/`, `custom_voices/`, `previews/`. Log nằm trong `~/Library/Logs/ToolVoiceMac/`. Nâng cấp gói ứng dụng không xóa các thư mục này. Cache model dùng Hugging Face cache. Không commit API key, file giọng cá nhân hoặc job.

OmniVoice tổng hợp trên máy sau khi model đã tải. Edge TTS gửi văn bản tới dịch vụ Microsoft; Gemini gửi nội dung tương ứng tới Google. Video Dubbing cần Gemini cho bước dịch. Server chỉ bind loopback; không được thiết kế để công khai trực tiếp ra Internet.

Giấy phép của trọng số và thư viện bên thứ ba được áp dụng riêng; xem [THIRD_PARTY.md](THIRD_PARTY.md). Repository đang chuẩn bị phát hành; việc public được thực hiện sau khi xác nhận giấy phép mã nguồn.

## Phát triển

```bash
uv venv --python 3.12.14
uv pip install --require-hashes -r requirements-macos.lock
uv pip install --no-deps -e .
uv pip install 'pytest>=8' 'pytest-asyncio>=0.24' 'httpx>=0.27'
.venv/bin/python -m pytest
.venv/bin/python -m toolvoice
```

Các kernel tối ưu có kiểm tra tương thích và số học lúc nạp. Không nâng torch/transformers hoặc đổi revision model tùy ý rồi coi kết quả kiểm chứng cũ vẫn còn hiệu lực. Công cụ benchmark và CLI dành cho phát triển được giữ riêng khỏi lệnh mở ứng dụng.
