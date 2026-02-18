# 🚀 Hướng Dẫn Upload Package VCDA-Net Lên GitHub

Bạn đã đổi tên package thành công sang `vcda-net` (Voxel Cluster Density Attention Network).

⚠️ **LƯU Ý QUAN TRỌNG**: Do bạn đã đổi tên toàn bộ cấu trúc thư mục, Git sẽ nhận diện đây là thay đổi lớn (xóa file cũ, thêm file mới).

---

## 1. Cập Nhật Git Local

Hãy chạy các lệnh sau để Git ghi nhận sự thay đổi tên này:

```bash
# 1. Thêm tất cả thay đổi (bao gồm việc đổi tên)
git add .

# 2. Commit thay đổi
git commit -m "Refactor: Rename package to VCDA-Net (Voxel Cluster Density Attention Network)"

# 3. Đổi tên nhánh chính (nếu chưa làm)
git branch -M main
```

---

## 2. Push Lên GitHub

Nếu bạn đã add remote `origin` trước đó, chỉ cần push:

```bash
git push -u origin main
```

Nếu gặp lỗi `Author identity unknown` hoặc cần nhập token, hãy dùng lệnh sau (thay Token của bạn vào):

```bash
git push https://USERNAME:YOUR_TOKEN@github.com/USERNAME/REPO_NAME.git main
```

---

## 3. Kiểm Tra Kết Quả

Sau khi push, trên GitHub bạn sẽ thấy:
- Thư mục `src/vcda_net`.
- File `train_vcda_net.sh`.
- Class `VCDANet` thay vì tên cũ.
- Tài liệu đã được cập nhật thành VCDA-Net.
