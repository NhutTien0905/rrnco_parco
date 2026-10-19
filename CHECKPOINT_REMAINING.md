# Checkpoint — Reviewer concerns chưa làm được (2026-10-18)

## Đã fix trong `paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex`
- R1.1/R2.2 feasibility: Eq per-trip + depot replenish
- R1.2/R2.4 gap: Table I 0.21/2.5% vs Table V 0.13% tách scale, define ΔG
- R1.3 Table VI +7 agents, Table VII Total Visits
- R1.4/R2.3/R3.2 Lemma13 rank, Lemma complexity O(N^2), Theorem22 D4, Th26/27 → Proposition/Remark
- R1.5 speedup Table IV steps 48.1 + dual protocol
- R3.1 ablation fusion 3 variants
- R3.5 neutral 3 đoạn chính

## Chưa làm được — cần làm tiếp

### 1) R2.1 Innovation hạn chế (Reviewer 2.1 + AE)
- Hiện tại: chỉ hạ claim "integration of existing RRNCO+PARCO" (Abstract/Conclusion).
- Chưa có: module/loss mới chứng minh novelty.
- Việc cần làm:
  - Option A (nhanh): reframe contribution =「systematic integration + real-world multi-agent benchmark + fusion empirical study」, thêm 1 ablations nhỏ mang tính novelty (ví dụ gate fusion learnable vs concat).
  - Option B (mạnh): thêm 1 algorithmic mới thực sự (ví dụ: heterogeneity-aware NAB hoặc speed-conditioned bias) + 1 table so sánh.
- File liên quan: `paper/.../tii-articles-template.tex:80-90` (contributions), `hybrid/models/encoder.py:_fuse`, `hybrid/models/nn/attn_freenet.py`

### 2) R3.4 Baselines thiếu (Reviewer 3.4 + AE) — BLOCKER nếu resubmit
- Hiện tại: mới thêm dòng `Classical HFVRP (planned)` (paper:668), chưa có số.
- Việc cần làm:
  - Chạy thêm 2-3 classical baselines cùng protocol N=100 M=7 60s: ALNS / HGS-CVRP / LKH3 (hoặc Google OR-Tools với time limit khác).
  - Cập nhật Table I thêm cột baselines, báo cáo gap vs hybrid.
  - Nếu không chạy được: đổi thành limitation rõ ràng + trích số published gần nhất làm proxy, nhưng rủi ro vẫn bị reject.
- File liên quan: `hybrid/configs/`, `hybrid/test.py`, `paper/.../tii-articles-template.tex:664`

### 3) R3.5 Language còn sót (nhẹ)
- Đã fix caption/3 đoạn, còn rải rác superlative.
- Việc cần làm: `grep -i "exceptional|outstanding|remarkable|superior|state-of-the-art|significantly improves"` toàn file và hạ tông.

## Checkpoint để resume
- Branch: `feat/optimize-fairness`
- Paper file gitignored (`paper/` in .gitignore) — edits chỉ local, cần `git add -f` để commit.
- Quyết định cần trước khi code tiếp: chọn Option A hay B cho R2.1, và có chạy baseline R3.4 không.

## Lệnh resume
```bash
cat CHECKPOINT_REMAINING.md
grep -n "planned\|TODO\|FIXME" paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex
```
