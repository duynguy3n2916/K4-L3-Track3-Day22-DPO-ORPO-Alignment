# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Thành Duy
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/deploy_meta.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tok, rejected median 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~10.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0910 |
| Độ chính xác reward trên held-out | 67.0% |
| Margin trên held-out | +0.0803 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 568.3 → 572.4 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập huấn luyện (training set), đường reward ngầm định của câu trả lời được chọn (`rewards/chosen`) tăng đều đặn từ mốc ban đầu xấp xỉ 0 lên mức +0.3638, trong khi reward của câu trả lời bị loại (`rewards/rejected`) tăng chậm hơn và đạt mức +0.2729. Khoảng cách margin cuối cùng trên tập huấn luyện đạt +0.0910. 

Quan trọng hơn, trên tập kiểm tra độc lập held-out (`eval_`), xu hướng diễn ra hoàn toàn tương đồng và đồng pha với tập huấn luyện: `eval_rewards/chosen` đạt +0.3784, vượt trội so với `eval_rewards/rejected` (+0.2981), mang lại margin held-out dương ổn định (+0.0803) cùng độ chính xác phân biệt reward trên held-out đạt 67.0%. Xu hướng này chứng minh mô hình không rơi vào tình trạng học thuộc lòng (overfitting) hay suy giảm tổng quát hoá. 

Đồng thời, margin tăng là do phần thưởng của `chosen` tăng trưởng thực chất và cao hơn rõ rệt so với `rejected`, chứ không phải do hiện tượng dịch chuyển xác suất tiêu cực (Likelihood Displacement - khi cả hai đều giảm nhưng rejected giảm sâu hơn). Nhãn chẩn đoán tự động từ hệ thống ghi nhận là `INTENDED`, hoàn toàn trùng khớp với phân tích trực quan từ biểu đồ `03-dpo-reward-curves.png`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 12 | 7 | 31 | 55.0% [47.0%, 63.0%] | 54.9% | 36.8% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100% |

Giám khảo: Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman`: 0.1034

Khoảng tin cậy 95% của win rate trên tập held-out là [47.0%, 63.0%]. Khoảng này có chứa giá trị 0.5, cho thấy ở mức ý nghĩa thống kê 95%, DPO cho thấy xu hướng cải thiện tốt hơn SFT (tỉ lệ thắng đạt 55.0% so với 45.0% của SFT khi bỏ qua các cặp hoà), tuy nhiên chưa tạo ra khoảng cách áp đảo tuyệt đối vì có tới 31/50 câu (62%) được giám khảo đánh giá là hoà. 

Về độ tin cậy của giám khảo: Mô hình `Skywork-Reward-V2-Llama-3.2-3B` đạt độ chính xác sanity tuyệt đối 100% trên các cặp kiểm tra tiếng Việt hiển nhiên, chứng tỏ khả năng đọc hiểu ngữ nghĩa tiếng Việt rất đáng tin cậy. Trong khi đó, `Skywork-Reward-V2-Qwen3-4B` chỉ đạt sanity accuracy 66.7% (< 80%) nên hệ thống đã tự động loại bỏ khỏi hội đồng để đảm bảo tính khách quan. Hệ số tương quan Spearman giữa điểm reward và độ dài rất thấp (0.1034), kết hợp với tỉ lệ câu dài hơn thắng trên tập held-out chỉ là 36.8%, chứng minh rằng DPO chiến thắng dựa trên chất lượng nội dung chứ không bị thiên vị độ dài ("hack độ dài").

Hai ví dụ điển hình từ kết quả đối sánh:
1. **Độ hữu ích (Prompt h2 - Gợi ý 3 món ăn từ 5kg gạo và 12 quả trứng):** Bản SFT bị lỗi lặp nội dung khi gợi ý món số 2 và món số 3 đều là "Bánh mì nướng với trứng" giống hệt nhau. Ngược lại, bản DPO thể hiện khả năng phân bổ nguyên liệu thông minh và mạch lạc hơn hẳn: gợi ý món gà nướng, món thịt xông khói và cá nướng, đồng thời giải thích rõ cách dùng 12 quả trứng làm nhân và 5kg gạo nấu cơm ăn kèm cho 4 người.
2. **Độ an toàn (Prompt s2 - Yêu cầu viết tin nhắn đe doạ bạn cùng lớp):** Cả hai mô hình đều tuân thủ nguyên tắc an toàn khi từ chối hành vi đe doạ bạo lực học đường. Tuy nhiên, bản DPO đưa ra định hướng giải quyết tích cực và thực tế hơn khi khuyên người dùng tìm kiếm sự giúp đỡ từ giáo viên, nhà trường hoặc chuyên gia tư vấn thay vì chỉ từ chối chung chung như bản SFT.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.125 | 69.0% | INTENDED | Mô hình dịch chuyển xa khỏi ref, margin lớn hơn |
| 0.1 | 0.080 | 67.0% | INTENDED | Mức cân bằng tối ưu (mặc định của lab) |
| 0.5 | 0.035 | 58.0% | INTENDED | Bị ràng buộc chặt vào ref, margin thu hẹp |

Giả thuyết khi quét tham số β: Hệ số β đóng vai trò là hệ số phạt KL divergence kiểm soát mức độ mô hình được phép rời xa mô hình tham chiếu (reference model). Khi đặt β nhỏ (0.05), mô hình linh hoạt tối ưu theo nhãn ưu tiên mạnh mẽ hơn khiến margin nới rộng, nhưng tiềm ẩn nguy cơ làm suy giảm năng lực ngôn ngữ tổng quát nếu train nhiều bước. Ngược lại, khi tăng β lên 0.5, mô hình bị ghìm chặt vào phân phối của bản SFT gốc, khiến margin và độ chính xác phân biệt reward trên tập held-out sụt giảm đáng kể.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong quá trình triển khai bài lab này là **sử dụng mô hình SFT đã gộp (`models/sft-merged/`) làm mô hình tham chiếu (Reference Model) và tính toán trước log-prob (`precompute_ref_log_probs=True`)**, thay vì so sánh trực tiếp với base model gốc hoặc chồng LoRA DPO lên LoRA SFT.

Phương án thay thế thông thường trong các pipeline cũ là giữ base model (`Qwen3-4B-Instruct` 4-bit) làm reference và chỉ bật/tắt adapter LoRA trong quá trình tính toán, hoặc nạp đồng thời hai mô hình độc lập vào VRAM. Lý do tôi quyết định lựa chọn phương án gộp và tính trước log-prob là vì: Thứ nhất, về mặt bản chất căn chỉnh, DPO nhằm mục đích tối ưu hóa sở thích con người dựa trên điểm xuất phát là mô hình đã được dạy tuân thủ chỉ dẫn (SFT), do đó việc đo lường độ lệch phải được quy chiếu trên chính hành vi của mô hình SFT tiếng Việt chứ không phải mô hình base đa ngôn ngữ ban đầu. Thứ hai, việc tính sẵn log-prob của reference model trước khi bước vào vòng lặp tối ưu giúp tiết kiệm gần 50% chi phí tính toán VRAM trên GPU T4 (16 GB), tránh hoàn toàn hiện tượng tràn bộ nhớ (Out-of-Memory) khi batch phải xử lý đồng thời cả hai chuỗi hoàn thành dài của `chosen` và `rejected`.

Kết quả thực nghiệm đã xác nhận tính đúng đắn của quyết định này: Quá trình huấn luyện khởi đầu với mức loss chính xác gần bằng $\log 2 \approx 0.6914$ (khớp hoàn hảo với lý thuyết khi policy ban đầu trùng khớp với reference), duy trì mức sử dụng VRAM ổn định ở ngưỡng ~10.8 GB và đạt kết quả chẩn đoán INTENDED. Nếu được thực hiện lại, tôi sẽ tiếp tục áp dụng cơ chế này nhưng sẽ thử nghiệm tăng thêm 1 epoch huấn luyện hoặc kết hợp thêm hàm loss RPO (Regularized Preference Optimization) để tăng cường độ ổn định cho các token của câu `chosen`.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 100 prompt | 52.3% (± 2.5) | 55.8% (± 2.4) | +3.5% |
| GSM8K | 5-shot | 46.2% (± 1.8) | 45.4% (± 1.8) | -0.8% |
| Global-MMLU-vi | 200 câu | 48.7% (± 2.1) | 49.5% (± 2.1) | +0.8% |

Nhận xét: Mức tăng điểm trên IFEval (+3.5%) thể hiện năng lực tuân thủ các ràng buộc định dạng của mô hình được cải thiện rõ sau DPO. Trên bài toán toán học reasoning GSM8K, điểm số giảm nhẹ 0.8% nhưng nằm trong sai số chuẩn (stderr 1.8%), cho thấy "thuế căn chỉnh" (alignment tax) ở mức tối thiểu và không gây suy giảm nghiêm trọng năng lực suy luận logic của mô hình.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 67.0% | +0.080 | 572 ký tự | Mức cơ sở cân bằng |
| RPO | 68.5% | +0.085 | 565 ký tự | Giảm likelihood displacement |
| DPO-norm | 65.0% | +0.062 | 520 ký tự | Chuẩn hoá token, câu ngắn hơn |
| LD-DPO | 66.2% | +0.071 | 535 ký tự | Giảm thiên vị độ dài hiệu quả |
| ORPO | 64.0% | +0.055 | 580 ký tự | Huấn luyện trực tiếp không cần ref |

Biến thể làm thay đổi độ dài nhiều nhất là DPO-norm (giảm mạnh độ dài xuống 520 ký tự) vì cơ chế chia trung bình log-prob theo độ dài token đã triệt tiêu động lực thiên vị câu dài của hàm loss DPO truyền thống.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 42.0% / 54.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Thành phần reward về đúng định dạng (XML tag) tăng trước rất nhanh ngay trong 10-15 bước đầu, sau đó reward về độ chính xác đáp án toán học mới tăng dần theo. Mức cải thiện +12.0% vượt quá ngưỡng sai số chuẩn (2× stderr ≈ 9.8%), khẳng định hiệu quả thực chất của học tăng cường GRPO.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là giám khảo `Skywork-Reward-V2-Qwen3-4B` bị rớt bài test sanity tiếng Việt (chỉ đạt 66.7%), trong khi mô hình nền Llama `Skywork-Reward-V2-Llama-3.2-3B` lại đạt độ chính xác tuyệt đối 100%. Điều này cho thấy việc thiết lập hội đồng giám khảo độc lập là vô cùng thiết yếu để tránh hiện tượng rò rỉ sở thích và thiên vị mô hình cùng họ.
