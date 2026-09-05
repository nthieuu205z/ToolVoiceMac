# Performance benchmarks

Các benchmark trong thư mục này chỉ đo, không thay đổi cấu hình production và không gọi
pipeline thật nếu chưa được chỉ định rõ. Kết quả local nên lưu dưới `.hermes/perf/`.

## Quy tắc

- Không ghi API key, token, password, prompt, transcript hoặc đường dẫn file người dùng.
- Metadata nhạy cảm bị harness loại bỏ; exception chỉ lưu loại lỗi, không lưu nội dung lỗi.
- Tách cold-start và warm-run.
- Mỗi case nên chạy ít nhất hai lần; báo median/p95 và error count.
- Không claim GPU utilization trên MPS nếu hệ điều hành/driver không cung cấp chỉ số đó.

## Harness API

`performance_harness.measure()` đo wall time, throughput, RSS, CPU/system memory và MPS
allocated/driver memory khi PyTorch cung cấp API tương ứng.

`write_json_report()` ghi report schema version 1 theo cách atomic và lọc metadata trước khi
lưu.

Các benchmark thật sẽ được thêm theo từng task trong plan tối ưu GPU/concurrency.
