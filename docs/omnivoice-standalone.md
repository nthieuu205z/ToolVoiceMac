# OmniVoice CLI dành cho phát triển

`run_omnivoice.py` gọi `pipeline/omnivoice_runtime.py` trực tiếp từ checkout. Công cụ này không dùng backend, Job Queue, catalog giọng hay dữ liệu của ứng dụng. Để mở giao diện ToolVoiceMac thông thường, dùng `toolvoice` theo [README](../README.md).

Chuẩn bị môi trường phát triển theo README, rồi chạy từ thư mục repository:

```bash
.venv/bin/python run_omnivoice.py --help
```

## Chuẩn bị đầu vào

- `--input`: file JSON UTF-8 chứa danh sách từ 1 đến 64 chuỗi không rỗng, tổng tối đa 50.000 ký tự. Giữ nguyên thứ tự; runtime gửi danh sách trong một request, không tự chia lô.
- `--reference-audio`: file audio tham chiếu bạn có quyền sử dụng.
- `--reference-text`: file UTF-8 chứa lời tham chiếu khớp audio; CLI không tự nhận diện bằng ASR.
- `--language`: `vi-VN` hoặc `en-US`; các alias trong catalog ngôn ngữ cũng được chấp nhận.
- `--output-dir`: thư mục mới, có thư mục cha đã tồn tại. Không ghi đè đầu ra cũ.
- `--seed`: số nguyên từ 0 đến 4.294.967.295, mặc định 1234.
- `--checkpoint`: đường dẫn checkpoint local chứa thư mục `audio_tokenizer`, hoặc repo ID đã cache. Mặc định là `k2-fsa/OmniVoice`; CLI chỉ đọc cache, không tự tải model hay codec.
- `--settings`: file JSON tùy chọn theo `OmniVoiceSettings`. Mặc định 32 bước, guidance 2.0, speed 1.0, duration tự động; denoise, preprocess và postprocess bật. Duration thủ công chỉ dùng cho một chuỗi tối đa 1.000 ký tự, lớn hơn 0 và không quá 30 giây.
- `--optimization`: `none` mặc định hoặc `split-cfg`. Đây là lựa chọn riêng của CLI, không thay cấu hình server.

Ví dụ nội dung `texts.json`:

```json
["Xin chào.", "Đây là câu thứ hai."]
```

Chuẩn bị các file trong `inputs/` và thay đường dẫn checkpoint bên dưới bằng snapshot local cần kiểm tra. Các file ví dụ không được cung cấp sẵn:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python run_omnivoice.py \
  --input inputs/texts.json \
  --reference-audio inputs/reference.wav \
  --reference-text inputs/reference.txt \
  --checkpoint /path/to/cached/snapshot \
  --language vi-VN \
  --seed 1234 \
  --output-dir standalone-output-001
```

CLI không đọc `.env`, không tự tìm giọng trong Application Support và không tự áp dụng revision đã ghim của ứng dụng. Để so sánh cùng checkpoint, truyền đường dẫn snapshot cụ thể qua `--checkpoint`.

## Đầu ra và hành vi

CLI tạo `0000.wav`, `0001.wav`, … theo thứ tự đầu vào, cùng `metadata.json`. Audio là WAV mono 24 kHz, PCM 16-bit. CLI này không xuất MP3. Metadata gồm số file, sample rate, thời gian từng giai đoạn và thông tin runtime đã lọc; không chứa nội dung text, transcript hoặc đường dẫn giọng tham chiếu.

Model được nạp qua CPU rồi chuyển sang MPS/float16; codec chạy trên CPU. Một phiên runtime nạp model và tạo prompt một lần, từ chối hai lượt generate đồng thời. Mỗi lần gọi CLI là một phiên mới. Không có retry, đổi provider hoặc giảm batch ngầm.

CLI kiểm tra toàn bộ output trước khi công bố thư mục bằng thao tác atomic dành cho macOS. Thành công trả mã 0; lỗi trả mã 2 với loại lỗi đã lược bỏ chi tiết nhạy cảm; ngắt bằng Ctrl+C trả mã 130. CLI không tự dừng server, không đặt deadline hay giới hạn bộ nhớ và không hủy giữa từng model step. Tránh chạy song song với job thật đang dùng GPU.

## Quan hệ với runtime ứng dụng

`none` giữ đường upstream gốc; `split-cfg` dùng subclass với kiểm tra source tương thích trước khi nạp. CLI chỉ hỗ trợ hai mode này và codec CPU. Tăng số mục trong request không bảo đảm tăng tốc; giới hạn 64 mục là giới hạn đầu vào, không phải khuyến nghị batch cho máy cụ thể.

Ứng dụng hiện dùng `split-cfg-rms-gqa-rope`, model MPS/FP16, codec MPS và batch tối đa 2; xem [runtime ứng dụng](RUNTIME.md). Không dùng thời gian đo của CLI độc lập để suy ra hiệu năng Job Queue, video hoặc project dài. Muốn kiểm tra mode server thực sự đã nạp, xem `GET /api/omnivoice/runtime`; chỉ giá trị cấu hình yêu cầu hoặc HTTP 200 chưa đủ chứng minh model sẵn sàng.
