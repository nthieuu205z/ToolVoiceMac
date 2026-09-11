# Runtime Mac đã kiểm chứng

ToolVoiceMac 0.2.0 giữ đường suy luận đang dùng trên M1 Max 64 GB:

| Thành phần | Cấu hình |
|---|---|
| Python | 3.12.14 arm64 |
| OmniVoice | 0.2.1 |
| torch / torchaudio | 2.13.0 / 2.11.0 |
| transformers | 5.16.1 |
| Model và codec | MPS, model FP16 |
| Optimization | split-cfg-rms-gqa-rope |
| Batch | tối đa 2 |
| Sinh giọng | 32 bước, guidance 2.0, speed 1.0 |
| Whisper | small/int8 CPU, batch tối đa 8 |

`requirements-macos.lock` khóa phiên bản và hash thư viện. Bộ máy đã cài thực tế được ghi ở `release/CURRENT_RUNTIME.json`; trạng thái trong file là ảnh chụp tại thời điểm kiểm tra, không phải trạng thái live của máy người đọc.

## Tối ưu đang dùng

Split-CFG chạy riêng hai nhánh điều kiện. Native RMSNorm và GQA giảm phép tính/tensor trung gian. Metal RoPE gộp phép xoay Q/K, giữ ranh giới làm tròn half của đường trước. Các helper chỉ bật cho backbone, kiểu dữ liệu và source đã kiểm chứng; không sửa thư viện site-packages toàn cục.

Lượt đối chiếu trước phát hành của riêng Metal RoPE so với split-CFG + RMSNorm + GQA giảm thời gian khoảng 18,7% trên bài tiếng Anh 18 đoạn và 19–22% trên các mẫu tiếng Việt. Token, RNG và audio trùng từng byte trên các mẫu đó. Các tối ưu lịch sử khác có thể thay đổi số học; không suy kết luận bit-exact này cho mọi thay đổi, mọi đầu vào hoặc thư viện tương lai.

Thử nghiệm gộp câu bị loại vì người dùng nhận thấy giảm độ tự nhiên. Thử nghiệm SiLU và Q/K normalization fusion chỉ tăng khoảng 0,65% / 1,02% trên hai đoạn đối chiếu và chưa được bật. Không hạ số bước hoặc lượng tử hóa để quảng cáo tốc độ.

## Vận hành

`toolvoice --doctor` kiểm tra phiên bản và FFmpeg mà không nạp model. API `/api/omnivoice/runtime` phải báo `state=ready`, `loaded_optimization=split-cfg-rms-gqa-rope`, `device=mps`, `codec_device=mps:0`, `batch_ceiling=2` mới xác nhận model đang chạy đúng cấu hình. HTTP/UI hoạt động chưa có nghĩa model đã nạp.

Có thể quay về mode `split-cfg-rms-gqa`, `split-cfg-rms`, `split-cfg` hoặc `none` trong `.env` và khởi động lại. Những mode này giữ để chẩn đoán/rollback, không tự đổi giữa chừng. Codec CPU vẫn là lựa chọn khôi phục có chủ đích; bản mặc định dùng MPS.

Đừng tăng batch hoặc bật đồng thời nhiều model chỉ vì thấy RAM còn trống. CPU/GPU dùng bộ nhớ hợp nhất; tốc độ còn phụ thuộc hình dạng tensor, padding, đồng bộ và tải máy. Các kết quả là phép đo trên phần cứng đã nêu, không phải cam kết tốc độ cho mọi Mac.
