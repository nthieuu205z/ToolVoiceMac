# Cài đặt ToolVoiceMac trên Mac

Dành cho Apple Silicon, macOS 14 trở lên. Xem [README](README.md) để lấy lệnh cài đúng tag phát hành; không chạy URL của tag chưa được xuất bản.

1. Cài Homebrew nếu chưa có và bạn cần cài FFmpeg: https://brew.sh .
2. Chạy installer của bản phát hành. Installer dùng uv để quản lý Python 3.12.14 và môi trường riêng, giữ đúng bộ thư viện trong `requirements-macos.lock`.
3. Mở Terminal mới, gõ `toolvoice`. Nếu thiếu PATH, chạy `uv tool update-shell` rồi mở Terminal mới.
4. Tải model trong giao diện. Nhập Gemini key ở Workspace nếu cần dịch video; không cần key cho clone TTS local.

## Lệnh

```bash
toolvoice
toolvoice --doctor
toolvoice --no-browser
toolvoice --port 8001
```

Tool chỉ dùng `127.0.0.1`. Nếu cổng thuộc ứng dụng khác, launcher báo lỗi và không tắt ứng dụng đó. Nếu cùng kho dữ liệu đã có server, launcher mở lại server đó thay vì chạy thêm model.

## Dữ liệu

- `~/Library/Application Support/ToolVoiceMac/.env`: thiết lập và key cá nhân.
- `~/Library/Application Support/ToolVoiceMac/jobs/`: lịch sử, đầu vào và kết quả.
- `~/Library/Application Support/ToolVoiceMac/custom_voices/`: giọng và lời tham chiếu.
- `~/Library/Application Support/ToolVoiceMac/previews/`: audio nghe thử phát sinh.
- `~/Library/Logs/ToolVoiceMac/toolvoice.log`: log ứng dụng.

Dùng `--data-dir` hoặc `TOOLVOICE_DATA_DIR` nếu muốn vị trí khác. Có thể đặt `TOOLVOICE_LOG_DIR` cho log. Hai bản dữ liệu cùng tên job/voice không được ghi đè để gộp; sao lưu trước khi chuyển từ checkout cũ. Dừng server trước khi sao chép kho dữ liệu.

Mẫu cấu hình nằm ở [.env.example](.env.example). Không chép key vào file mẫu rồi đẩy lên Git.

## Xử lý lỗi

- Thiếu FFmpeg: `brew install ffmpeg`, sau đó `toolvoice --doctor`.
- Sai phiên bản kernel/model: cài lại đúng bản phát hành; xem log. Cấu hình tối ưu phụ thuộc bộ thư viện và revision đã ghim.
- Model chưa tải: dùng giao diện quản lý model; không cần tải lại nếu đã có trong cache hợp lệ.
- Dùng checkout để phát triển: làm theo phần Phát triển trong README; `ChayTool.command` gọi cùng launcher qua môi trường checkout hoặc lệnh đã cài.

## Sử dụng và dịch vụ mạng

Text → Voice hỗ trợ tiếng Việt (`vi-VN`) và tiếng Anh (`en-US`), xuất WAV/MP3. Job đơn nhận tối đa 50.000 ký tự; Project nhận tối đa 200.000 ký tự, chia phần và tải ZIP theo thứ tự. Voice Lab chọn riêng ngôn ngữ đầu ra của từng giọng clone.

OmniVoice clone trên máy sau khi tải model. Edge TTS gửi văn bản tới Microsoft; provider Gemini gửi nội dung tương ứng tới Google. Video Dubbing cần Gemini API key cho bước dịch; nhập key tại Workspace.
