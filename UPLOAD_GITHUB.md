# 🚀 Hướng Dẫn Upload Package Lên GitHub

Bạn đã chuẩn bị xong package `hybrid-saliency-v4`. Dưới đây là các bước để đưa code lên GitHub một cách an toàn và chuyên nghiệp.

---

## 1. Tạo Repository Trên GitHub (Quan Trọng)

1. Đăng nhập vào [GitHub](https://github.com/).
2. Nhấn dấu `+` ở góc trên bên phải → chọn **New repository**.
3. Điền thông tin:
   - **Repository name**: `hybrid-saliency-v4` (hoặc tên tùy ý).
   - **Description**: "Brain Age Prediction with Saliency Map-Enhanced Features and Gated Fusion".
   - **Public/Private**: Tùy bạn chọn (Nếu cần ẩn danh để review báo, hãy chọn **Private**).
4. ⚠️ **RẤT QUAN TRỌNG**:
   - **KHÔNG** tích chọn "Add a README file".
   - **KHÔNG** tích chọn "Add .gitignore".
   - **KHÔNG** chọn License.
   *(Lý do: Chúng ta đã tạo sẵn tất cả các file này ở local rồi. Nếu chọn sẽ gây xung đột khi push).*
5. Nhấn **Create repository**.

---

## 2. Chuẩn Bị Code Ở Local (Đã làm xong phần dọn dẹp)

Gói package của bạn đã được dọn dẹp sạch sẽ (đã xóa các thư mục log, output, cache).

Bây giờ, hãy mở terminal tại thư mục package và chạy các lệnh sau để khởi tạo Git:

```bash
# 1. Di chuyển vào thư mục package (nếu chưa ở đó)
cd /media/devin/WORK/devin/tien/src/brain_age_prediction/hybrid_saliency_v4_package

# 2. Khởi tạo Git
git init

# 3. Thêm tất cả file vào git
git add .

# 4. Commit phiên bản đầu tiên
git commit -m "Initial commit: Hybrid Saliency V4 Package (v4.0.2)"

# 5. Đổi tên nhánh chính thành 'main' (chuẩn mới của GitHub)
git branch -M main
```

---

## 3. Đẩy Code Lên GitHub

Sau khi tạo repo ở Bước 1, GitHub sẽ cung cấp cho bạn một đường link HTTPS hoặc SSH (ví dụ: `https://github.com/yourusername/hybrid-saliency-v4.git`).

Hãy copy đường link đó và chạy lệnh sau:

```bash
# Thay thế URL bên dưới bằng URL repo của bạn
git remote add origin https://github.com/USERNAME/hybrid-saliency-v4.git

# Đẩy code lên
git push -u origin main
```

---

## 4. Lưu Ý Về Xác Thực (Authentication)

Nếu bạn dùng **HTTPS** (`https://...`), khi `git push` hỏi mật khẩu:
- Từ tháng 8/2021, GitHub **không hỗ trợ mật khẩu tài khoản**.
- Bạn phải dùng **Personal Access Token (PAT)**.

**Cách tạo Token (nếu chưa có):**
1. Vào GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic).
2. Generate new token.
3. Chọn các quyền (scopes): tích vào `repo` (đủ quyền push code).
4. Copy chuỗi token (bắt đầu bằng `ghp_...`).
5. Dán chuỗi này vào terminal khi được hỏi Password.

(Nếu bạn đã cài đặt SSH key, hãy dùng link SSH `git@github.com:...` để không cần nhập mật khẩu/token).

---

## ✅ Kiểm Tra Sau Khi Upload

Truy cập lại trang repository trên GitHub, bạn sẽ thấy:
- File `README.md` hiển thị đẹp mắt.
- Mã nguồn nằm trong thư mục `src/`.
- Không có các file rác hay thư mục output lớn.
- Package sẵn sàng để người khác cài đặt bằng `pip install git+https://...`.
