# Thành phần bên thứ ba

Giấy phép mã nguồn ToolVoiceMac không thay thế giấy phép của model, thư viện, dịch vụ hay dữ liệu giọng người dùng.

- **OmniVoice 0.2.1:** mã nguồn upstream Apache-2.0. Các phần điều chỉnh từ upstream giữ thông báo bản quyền trong source. [Giấy phép upstream](https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE).
- **Trọng số k2-fsa/OmniVoice:** model card tại revision đã ghim `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` ghi CC-BY-NC do ràng buộc dữ liệu huấn luyện. Không coi giấy phép code là quyền dùng trọng số cho mục đích thương mại. [Model card của đúng revision](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md).
- **PyTorch, Transformers, faster-whisper và các dependency:** được cài riêng theo `requirements-macos.lock`; thông báo giấy phép nằm trong từng distribution.
- **FFmpeg/ffprobe:** cài riêng trên máy; giấy phép phụ thuộc bản build và các codec đi kèm.
- **Microsoft Edge TTS và Google Gemini:** dịch vụ mạng với điều khoản riêng; chỉ OmniVoice thực hiện clone local.

Repository và wheel không đóng gói trọng số model, API key, mẫu giọng cá nhân hoặc kết quả job. Model được tải từ nguồn upstream và dùng cache trên máy.
