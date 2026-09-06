# ToolVietSub

Công cụ đa ngôn ngữ `vi-VN` / `en-US` với hai workflow dùng chung một hàng đợi:

- **Video Dubbing:** tải video, chọn ngôn ngữ đích và giọng; nhận video đã thay track âm thanh cùng phụ đề `.srt` theo ngôn ngữ đã chọn.
- **Text → Voice:** nhập tối đa **50.000 ký tự**, chọn ngôn ngữ/giọng; nhận cả **WAV** và **MP3**.

Media, job metadata và Text → Voice input được xử lý/lưu cục bộ. Video Dubbing luôn cần
Gemini cho bước dịch; Text → Voice bằng Edge hoặc engine local không cần khóa Gemini.
`edge-tts` vẫn cần Internet, còn giọng nhân bản OmniVoice chạy offline sau khi tải model.

Mặc định xử lý cục bộ phần media; bước dịch gọi Gemini và `edge-tts` gọi dịch vụ giọng đọc
qua mạng:

| Bước | Chạy bằng | Ghi chú |
|---|---|---|
| Khung thời gian | ffmpeg `silencedetect` — trên máy | tìm vùng có tiếng nói |
| Nhận diện giọng nói | **Whisper trên máy** (GPU nếu có CUDA; macOS chạy CPU) | gộp lô CUDA ~9× nhanh hơn; lấy mốc TỪNG TỪ để cắt câu |
| Dịch | **Gemini Flash** (các lô chạy song song) | bước duy nhất cần mạng |
| Giọng đọc | **OmniVoice** trên máy (CUDA/MPS/CPU) / edge-tts qua mạng | không giới hạn lượt |

Mặc định `STT_PROVIDER=whisper` + `TTS_PROVIDER=edge`: không tốn token, không hạn mức API,
và trên máy có tăng tốc thì nhanh — video 19 phút xong **cả** pipeline trong ~2 phút. Chỉ dùng khóa
Gemini cho bước dịch.

> **Có card NVIDIA?** Whisper và OmniVoice tự dùng CUDA sau khi thử một phép tính thật. Cỡ lô **tự suy ra từ
> VRAM còn trống** (không ghim cứng), nên cùng một công thức tự ép tối đa mọi loại card. Trên
> Apple Silicon, OmniVoice thử Metal/MPS; Whisper vẫn chạy CPU vì CTranslate2 chưa có backend Metal.
> Card lớn chạy lô lớn hơn. Không có thiết bị gia tốc thì tự lùi về CPU. Windows + GPU: xem
> [HUONG_DAN_WINDOWS.md](HUONG_DAN_WINDOWS.md).

Vẫn đổi được sang `STT_PROVIDER=gemini` (8 luồng song song, tắt thinking) nếu muốn chép
lời chuẩn hơn với thuật ngữ và tên riêng — nhưng khi đó **mất mốc từng từ nên tắt việc
cắt câu** (xem "Đọc theo từng câu" bên dưới), pipeline lùi về đặt cả vùng tại một mốc.

### Chọn giọng đọc

Đổi `TTS_PROVIDER` trong `.env` để chọn giọng dựng sẵn:

| | Số giọng | Cần tải | Tốc độ (video 8,8 phút) | Ghi chú |
|---|---|---|---|---|
| `edge` (mặc định) | **14** | không | ~0,5 phút | 2 giọng Việt bản địa + 12 giọng đa ngôn ngữ; cần mạng |
| `gemini` | **10** | không | — | Trần ~100 lượt/ngày → video dài sẽ câm giữa chừng |

Mười hai giọng "Multilingual" của edge-tts mang nhãn `en-US`, `fr-FR`, `ko-KR`… nhưng đọc được tiếng Việt. Đây không phải suy đoán: mỗi giọng được cho đọc một câu tiếng Việt rồi bắt Whisper nghe lại — cả 12 đều được nhận là tiếng Việt với độ khớp 0,94–1,00. Vì edge-tts gọi endpoint của Microsoft, chế độ này cần Internet.

### Nhân bản giọng (OmniVoice)

Bấm **＋ Nhân bản giọng** cạnh tiêu đề "Chọn giọng đọc", đặt tên và tải một đoạn audio **3–8 giây** (một người nói, ít tạp âm). Giọng mới xuất hiện ngay trong danh sách; file nghe thử được tạo ở nền. Chỉ dùng giọng bạn có quyền sử dụng.

Giọng nhân bản được **định tuyến riêng** khỏi giọng dựng sẵn qua `CLONE_TTS_PROVIDER` (mặc định `omnivoice`): chọn một giọng nhân bản → đọc bằng engine này; chọn giọng dựng sẵn → vẫn dùng `TTS_PROVIDER`. Nhờ vậy có thể ghép **edge cho giọng dựng sẵn + OmniVoice cho giọng nhân bản**. Mỗi job dùng một giọng nên đây là một quyết định duy nhất cho cả job.

| `CLONE_TTS_PROVIDER` | Cách clone | Ghi chú |
|---|---|---|
| `CLONE_TTS_PROVIDER=omnivoice` (mặc định) | zero-shot đa ngôn ngữ | Model ~3,3 GB tải trên giao diện; **CC-BY-NC (phi thương mại)**. |
| `none` | — | Tắt tính năng nhân bản |

**Cài OmniVoice** — dùng extra inference của dự án:

```
uv pip install -e ".[dev,omnivoice]"
```

Model tự tải lần đầu, hoặc bấm nút tải trong mục "Model trên máy". `ref_text` (lời của clip mẫu) do Whisper chép **một lần** rồi nhớ trong file cạnh clip.

Vài điều đáng biết:
- OmniVoice không giữ giọng trong RAM; xóa giọng là sạch.
- Id giọng đã xóa **không bao giờ được cấp lại** (file `.tombstone` giữ chỗ).
- **License OmniVoice là CC-BY-NC** (phi thương mại) — dùng cá nhân thoải mái.

---

## Cài đặt

Hướng dẫn dưới đây dành cho macOS. **Windows (kèm bật GPU NVIDIA cho bước giọng đọc):
xem [HUONG_DAN_WINDOWS.md](HUONG_DAN_WINDOWS.md).**

**1. ffmpeg** (bắt buộc — dùng để tách, ghép, chỉnh tốc độ âm thanh)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

Nếu ffmpeg nằm ngoài `PATH`, khai báo `FFMPEG_BIN` và `FFPROBE_BIN` trong `.env`.

**2. Python 3.10+ và thư viện**

Dự án dùng [uv](https://docs.astral.sh/uv/). Nếu chưa có:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Rồi:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,omnivoice]"
```

**3. Khóa Gemini API** (chỉ bước dịch cần)

Lấy khóa miễn phí tại <https://aistudio.google.com/apikey>, rồi:

```bash
cp .env.example .env
# mở .env và điền GEMINI_API_KEY
```

Nếu khóa của bạn tạo trong **Google Cloud** thay vì AI Studio, Developer API sẽ trả `403 API_KEY_SERVICE_BLOCKED`. Khi đó đặt trong `.env`:

```
GEMINI_BACKEND=vertex
```

Cùng một khóa nhưng đi tới `aiplatform.googleapis.com` (Vertex AI Express Mode) thay vì `generativelanguage.googleapis.com`. Vertex không có `models.list()` và không phải model nào cũng tồn tại ở đó — `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro` thì có.

**4. Kiểm tra mọi thứ đã sẵn sàng**

```bash
./.venv/bin/python scripts/check_setup.py
```

Script này xác nhận ffmpeg, khóa API, và **kiểm tra tên model đã cấu hình có thật không** — Google đổi tên model khá thường xuyên, nếu sai nó sẽ in ra danh sách model đang khả dụng để bạn sửa `.env`.

---

## Chạy và sử dụng

```bash
./.venv/bin/uvicorn backend.main:app --port 8000
```

Mở <http://localhost:8000>. Trong **New Job**:

1. Chọn tab **Video Dubbing** để kéo thả video, chọn `vi-VN` hoặc `en-US`, chọn giọng và bắt đầu xử lý.
2. Chọn tab **Text → Voice** để nhập nội dung (tối đa **50.000 ký tự**), chọn ngôn ngữ/giọng, nghe giọng mẫu hoặc nghe thử chính nội dung rồi tạo job.
3. Cả hai workflow đi vào cùng **Job Queue**, dùng chung SSE, hủy/xóa, lịch sử khởi động lại và Execution Graph thích ứng.
4. Voice Lab có bộ chọn ngôn ngữ; demo cố định được tạo theo từng ngôn ngữ, còn nghe thử nội dung không lưu text thành artifact công khai.

Danh sách giọng tự lọc theo capability `supported_languages`; đổi ngôn ngữ sẽ giữ giọng nếu
tương thích, nếu không chọn giọng tương thích đầu tiên.

Tùy chọn — tạo file nghe thử giọng đọc để bấm nghe ngay trên giao diện (mỗi giọng tốn một lần gọi TTS, chỉ cần chạy một lần):

```bash
./.venv/bin/python scripts/generate_voice_previews.py
```

Hoặc chạy thẳng bằng dòng lệnh, không qua giao diện:

```bash
./.venv/bin/python scripts/run_pipeline_cli.py video.mp4 --voice Charon -v
```

---

## Cách hoạt động

Video Dubbing có bảy bước chạy tuần tự trong `pipeline/runner.py`:

| Bước | Việc làm |
|---|---|
| `extract` | ffmpeg tách audio ra WAV mono 16 kHz |
| `transcribe` | **ffmpeg** khoanh vùng có tiếng, **Whisper** chép lời + mốc từng từ → tách thành **từng CÂU** đặt đúng thời điểm |
| `translate` | Gemini dịch sang **ngôn ngữ đích đã chọn** (các lô chạy **song song**), giữ nguyên số dòng và thứ tự |
| `synthesize` | OmniVoice/edge-tts đọc **từng câu**, gộp lô trên GPU khi smoke test đã xác nhận |
| `subtitle` | Dựng `.srt`, cue bám theo thời lượng giọng đọc thật |
| `assemble` | Đặt từng đoạn vào đúng mốc thời gian trên nền im lặng dài bằng video |
| `mux` | ffmpeg ghép video gốc + track mới, **không** map audio gốc |

### ffmpeg lo thời gian, Gemini lo nội dung

Đây là quyết định thiết kế quan trọng nhất, và nó đến từ đo đạc chứ không phải trực giác.

Gemini có tham số `audio_timestamp`, nhưng nó **chỉ chạy trên Vertex AI**. Với khóa AI Studio (Developer API), khi bị hỏi mốc thời gian Gemini sẽ *bịa ra*. Đo trên 120 giây đầu của một video thật:

```
ffmpeg:  0.79–13.37   13.79–25.00   26.27–45.24   45.70–54.42
Gemini:  0.00–13.90   13.90–25.90   25.90–37.90   ... và một mốc kết thúc ở giây 200
```

Hai mốc đầu còn khớp, sau đó trôi dần, và cuối cùng nó trả về một segment kết thúc ở giây 200 của một đoạn audio chỉ dài 120 giây. Khoảng lặng luôn đúng bằng 0.00s — dấu hiệu rõ ràng là các mốc được suy ra chứ không được đo.

Nên pipeline dùng `silencedetect` của ffmpeg để tìm các vùng có tiếng nói thật, rồi gửi **từng vùng** cho model chép lời. Mốc vùng là thật, và văn bản chắc chắn thuộc đúng vùng đó.

### Đọc theo từng câu, đặt đúng thời điểm

ffmpeg khoanh vùng có tiếng, nhưng một vùng thường chứa **nhiều câu** (trần 12 giây). Đọc
cả khối rồi đặt tại một mốc khiến tiếng Việt lệch dần so với hình, và một câu tiếng Anh bị
vùng chẻ đôi sẽ đọc thành "Hôm ...(nghỉ)... nay". Nên với Whisper, pipeline bật
`word_timestamps` lấy mốc **từng từ**, rồi tách mỗi vùng thành **từng câu** theo dấu chấm
câu (`pipeline/whisper_stt.py::sentences_from_words`). Mỗi câu:

- là **một lượt đọc riêng** — model giọng nói vốn ổn định ở mức câu; nhồi nhiều câu
  làm một khối khiến nó thỉnh thoảng chèn khoảng lặng dài hoặc đọc bịa (đo thật: một khối
  4 câu ra lỗ hổng im lặng 4,72s ở giữa);
- được **đặt đúng thời điểm câu tiếng Anh** (mốc đầu câu) nên bám hình sát hơn hẳn;
- câu bị vùng chẻ đôi được **ghép lại** (mảnh chưa kết bằng dấu câu nối với mảnh sau, không
  vượt trần thời lượng).

ffmpeg vẫn lo **thời gian thô** (ràng buộc Whisper chỉ chép chỗ có tiếng), Whisper tinh mốc
câu bên trong — mốc căn chỉnh thật của nó, không phải bịa như Gemini. Bật/tắt qua
`SENTENCE_LEVEL_TIMING` (mặc định bật; `STT_PROVIDER=gemini` tự tắt vì Gemini không có mốc).

Đo trên cửa sổ 1:30–1:40 của một video thật, trước/sau khi đọc theo câu: lỗ hổng im lặng
dài nhất **4,72s → 0,86s**, tiếng Việt bám hình **75% → 85%**, số lỗ hổng ≥1,5s cả video
**39 → 11**.

### Vài điểm đáng biết

- **Audio gốc bị xóa hoàn toàn.** Lệnh mux chỉ map `0:v:0` và `1:a:0`. Có hẳn một test (`tests/test_mux_args.py`) canh để không ai vô tình thêm `-map 0:a`.
- **Video không bị encode lại** (`-c:v copy`) nên nhanh và không giảm chất lượng hình. Nếu container `.mp4` không chứa nổi codec gốc, hệ thống tự lùi về `.mkv`.
- **Phụ đề dựng SAU giọng đọc.** Giọng Việt thường đọc xong sớm hơn khung thời gian gốc; nếu trải cue theo khung thì các cue cuối rơi vào chỗ im lặng. Đo thực tế: 17% cue lệch khỏi tiếng nói → 0% sau khi bám theo thời lượng đọc thật.
- **Mỗi CÂU là một mốc neo đồng bộ.** Với Whisper, mỗi câu đặt đúng mốc câu tiếng Anh (mốc từng từ). Trần một vùng ffmpeg là `MAX_UTTERANCE_SECONDS` (mặc định 12) — nay chỉ dùng để chặn vùng quá dài *trước khi* tách câu, chứ không còn là hạt đồng bộ (xem "Đọc theo từng câu").
- **Tiếng Việt dài hơn khung → tăng tốc theo nấc; NGẮN hơn khung → kéo giãn nhẹ.** Dài hơn: (1) tăng tốc *nhẹ* (≤1,15× — dưới ngưỡng tai) về sát khung; (2) phần dư tràn vào khoảng lặng phía sau tới sát mốc câu kế tiếp; (3) hết chỗ mượn mới tăng tốc mạnh (tối đa `TTS_MAX_SPEEDUP`, mặc định **1,3×**), vẫn dư thì câu sau **lùi lại chờ** thay vì hai giọng đè nhau; câu ngắn được kéo giãn nhẹ (tối đa tới sàn `TTS_FILL_SLOWDOWN=0.9`, dài thêm ~11%, tai không nhận ra) để bám hình thay vì để im lặng cụt. Không bao giờ **cắt chữ**. Phụ đề luôn bám mốc phát thật.
- **Không đoạn nào bị bỏ rơi trong im lặng.** Đoạn nhận diện hỏng (dính 429 lúc 8 luồng dồn dập) được thử lại tuần tự sau khi cơn dồn request dịu; vẫn hỏng thì báo rõ trên màn kết quả kèm mốc thời gian, không lẳng lặng thiếu lời thoại.
- **Chống TTS "chạy hoang" + đọc lại lượt xấu.** Model tự hồi quy thỉnh thoảng bịa lời khi đầu vào quá ngắn; pipeline đặt trần độ dài, cắt kèm fade và có thể đọc lại lượt có khoảng lặng bất thường. Gemini TTS tính tiền theo lượt nên để tắt.
- **Lời thoại không được đưa trần vào TTS.** Gặp câu hỏi, model tưởng đó là câu lệnh và định trả lời (`"Model tried to generate text, but it should only be used for TTS"`). Pipeline bọc mỗi lượt trong một câu lệnh đọc nguyên văn — đã kiểm chứng là câu lệnh đó không bị đọc thành tiếng.
- **Số dòng dịch phải khớp tuyệt đối.** Lệch một dòng là lệch giờ toàn bộ phần sau, nên hệ thống thử lại một lần rồi báo lỗi thay vì xuất ra video sai tiếng. Prompt dịch cũng cấm lược ý: khung thời gian chỉ quyết định *cách diễn đạt*, không quyết định *lượng thông tin* — câu dài ra đã có cơ chế mượn khoảng lặng ở trên lo.
- **Nhiều job cùng lúc.** Video Dubbing và Text → Voice dùng chung hàng đợi; mỗi job có thẻ và tiến trình riêng. Tối đa `MAX_CONCURRENT_JOBS` (mặc định 2) job chạy đồng thời, job nộp thêm xếp hàng chờ. Trần đặt thấp có chủ ý: Whisper/OmniVoice bị khóa suy luận toàn cục và edge-tts bị trần 2 request đồng thời, nên job thứ ba chủ yếu chen hàng chứ không nhanh thêm.
- **Mở rộng ngôn ngữ:** thêm registry trong `pipeline/languages.py`, khai báo capability cho giọng/provider, thêm preview text và contract tests cho code canonical. Không thêm option UI đơn lẻ vì backend registry là nguồn sự thật.
- **Danh sách job sống sót qua mọi thứ.** Mỗi job ghi `job.json` vào thư mục của nó; đóng tab, quay lại trang chủ, hay khởi động lại server đều thấy nguyên danh sách và tải lại được kết quả cũ. Job đang chạy dở lúc server chết được đánh dấu lỗi kèm lời nhắn "hãy chạy lại" — không bao giờ hiện "đang chạy" ma.
- **Hủy được giữa chừng.** Bấm **Hủy** trên thẻ job (hoặc `POST /api/jobs/{id}/cancel`). Việc hủy là *hợp tác*: máy chủ không giết thread giữa chừng — lúc đó ffmpeg có thể đang ghi file và ONNX đang chạy — mà đặt cờ rồi để pipeline tự dừng ở mốc an toàn gần nhất. Đo thực tế trên video 19 phút: bấm hủy ở đoạn 2/64, dừng hẳn sau **4 giây**. Job còn xếp hàng thì hủy tức thì.

---

## Hạn mức và đánh đổi

### Vì sao mặc định không dùng Gemini TTS

**Bản miễn phí của Gemini cho khoảng 100 lượt gọi TTS mỗi ngày, cho mỗi model** — đo được, không phải suy đoán (`GenerateRequestsPerDayPerProjectPerModel = 100`). Pipeline gọi TTS một lần cho mỗi lượt phát ngôn: video 8,8 phút cần 32 lượt, video 18,9 phút cần 64 lượt. Tức là **khoảng 30 phút video mỗi ngày**, cộng dồn. Lưu ý: cấu hình mặc định dùng `edge-tts`, không dùng Gemini TTS; edge-tts vẫn cần mạng.

`edge-tts` không có trần đó nhưng cần mạng. Đánh đổi:

- **Chỉ 2 giọng tiếng Việt** thay vì 30 giọng của Gemini (ngoài các giọng multilingual được liệt kê trong giao diện).
- Là endpoint đọc-thành-tiếng của trình duyệt Edge, dùng theo cách **không chính thức**. Microsoft không cam kết gì; nó có thể ngừng chạy bất cứ lúc nào. Khi đó đổi `TTS_PROVIDER=gemini` trong `.env` là quay lại được ngay.
- **Nó bóp tần suất.** Đo thực tế: 6 luồng song song → 25/32 request thành công; 2 luồng kèm thử lại → 10/10. Vì vậy `TTS_WORKERS=2` và `EDGE_TTS_ATTEMPTS=5`.

### Whisper hay Gemini ở bước nhận diện

`STT_PROVIDER=whisper` (**mặc định**): miễn phí, không hạn mức, chạy offline, và **cho mốc
từng từ để cắt câu**. Trên GPU NVIDIA thì nhanh — gộp cả lô vùng trong một lượt gọi (cỡ lô
tự suy từ VRAM trống):

    từng vùng một   : 67 giây   (GPU 34%)
    gộp lô (GPU)    :  7 giây               ← ~9× nhanh hơn

Video 19 phút chép lời xong ~13 giây khi model đã nạp; server nạp sẵn Whisper lúc boot để
job đầu khỏi chờ ~11 giây tải model. Không có GPU thì tự lùi về CPU (chậm hơn nhiều). Lần
chạy đầu tải model (~484 MB). Muốn chép chính xác hơn: `WHISPER_MODEL=medium` (chậm hơn).

`STT_PROVIDER=gemini` (khi chấp nhận trả phí): mỗi vùng gửi thẳng cho Gemini, 8 luồng song
song, tắt thinking (đo gemini-3.5-flash/Vertex: 4,2–18,9s/request → ~2,3s khi tắt). Chép
chuẩn hơn với thuật ngữ và tên riêng, nhưng **không có mốc từng từ nên tắt việc cắt câu**.

Lưu ý về thời gian: ffmpeg cho ranh giới **vùng** chính xác (khoanh chỗ có tiếng); Whisper
tinh **mốc câu** bên trong bằng mốc từng từ. Gemini trên Developer API không đo được mốc,
chỉ chép chữ.

### Nếu vẫn muốn dùng Gemini TTS

Đặt `TTS_PROVIDER=gemini`. Khi chạm trần, app **không im lặng giả vờ thành công**: nó dừng thử lại ngay (Google bảo đợi hơn 4 tiếng), báo đỏ ở màn hoàn tất, cho biết bao nhiêu lượt bị bỏ trống, và vẫn cho tải video + phụ đề đầy đủ về. Muốn bỏ trần thì bật thanh toán trong Google Cloud — khi đó **giọng đọc là khoản tốn nhất**, tra giá tại <https://ai.google.dev/gemini-api/docs/pricing>.

---

## Hiệu năng (GPU)

OmniVoice chạy batch trên GPU khi smoke test của platform đã pass; mọi cỡ lô tự suy từ VRAM
trống ở mức an toàn, không ghim cứng:

- **Nhận diện (Whisper):** gộp cả lô vùng trên CUDA (~9× so với từng vùng một), nạp sẵn model
  lúc boot. Trên macOS Whisper chạy CPU; `word_timestamps` (để cắt câu) chỉ tốn thêm ~1,5s.
- **Dịch (Gemini):** các lô chạy **song song** (`TRANSLATE_WORKERS`, mặc định 6); ngữ cảnh
  lấy từ lời gốc nên lô nào cũng độc lập. Đo: 63,7s → 33,6s.
- **Đọc giọng nhân bản (OmniVoice):** gộp nhiều câu bằng một lượt `generate(list[str])` trên GPU.
  Cỡ lô mặc định cạp ở khoảng 8; `OMNIVOICE_NUM_STEP` đổi tốc độ⇄chất lượng. MPS dùng
  đường nạp CPU rồi chuyển model sang Metal để tránh crash khi accelerate dispatch trực tiếp.

Cờ chỉnh trong `.env` (đều có mặc định an toàn):

| Cờ | Mặc định | Việc |
|---|---|---|
| `SENTENCE_LEVEL_TIMING` | `true` | Đọc theo từng câu theo mốc từng từ (chỉ Whisper) |
| `TTS_FILL_SLOWDOWN` | `0.9` | Sàn kéo giãn để câu ngắn bám hình; `1.0` = tắt |
| `TTS_MAX_SPEEDUP` | `1.3` | Trần tăng tốc câu dài |
| `TRANSLATE_WORKERS` | `6` | Số lô dịch song song (hạ nếu Gemini free tier bị 429) |
| `CLONE_TTS_PROVIDER` | `omnivoice` | Engine đọc giọng nhân bản: `omnivoice` / `none` |
| `OMNIVOICE_NUM_STEP` | `32` | Số bước sinh OmniVoice; hạ 16–24 để nhanh hơn |
| `OMNIVOICE_BATCH_SIZE` | `0` (tự) | Ép cứng cỡ lô OmniVoice nếu muốn |

---

## Test

```bash
./.venv/bin/python -m pytest
```

Toàn bộ test dùng một Gemini giả (`tests/conftest.py`) — **không gọi API thật, không tốn token**, và không cần ffmpeg.

Provider smoke là opt-in vì gọi mạng/model thật:

```bash
RUN_PROVIDER_SMOKE=1 ./.venv/bin/python -m pytest -q -m provider_smoke
```

Edge English chạy khi có mạng. Gemini chỉ chạy khi `GEMINI_API_KEY` đã cấu hình. OmniVoice
clone chỉ chạy khi đặt `PROVIDER_SMOKE_CLONE_VOICE_ID` thành ID của một giọng local mà bạn
có quyền sử dụng; thiếu precondition nào thì test tương ứng báo `SKIP`, không được tính là pass.

---

## Cấu trúc

```
backend/     FastAPI: config, quản lý job, các route
pipeline/    7 bước xử lý + adapter Gemini (gemini.py là nơi duy nhất gọi API thật)
web/static/  Giao diện: HTML/CSS/JS thuần, không build step
scripts/     check_setup, chạy CLI, tạo file nghe thử giọng
tests/       Test đơn vị với Gemini giả
jobs/        File tạm và kết quả (đã gitignore)
```
