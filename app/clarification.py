"""Web-only clarification menus. Choices describe symptoms, never diagnoses."""

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip()


def symptom_menu(question: str):
    text = normalize(question)
    # Educational questions do not describe an unclear personal symptom.
    educational = (
        "nguyen nhan", "la gi", "dau hieu nao", "nhung dau hieu", "trieu chung cua",
        "phong ngua", "phong tranh", "nen an", "che do", "co vai tro", "lich kham",
    )
    if any(phrase in text for phrase in educational):
        return None

    groups = [
        (r"\bdau bung\b", "Bạn đau ở vùng nào?", [
            "Đau vùng bụng trên", "Đau vùng bụng dưới", "Đau một bên bụng", "Đau khắp bụng / khó xác định vị trí",
        ], r"\b(bung tren|bung duoi|ben trai|ben phai|mot ben|quanh ron|thuong vi|ha vi|khap bung)\b"),
        (r"\b(dau|nhuc)\b(?!\s+hieu)", "Bạn cảm thấy đau ở đâu?", [
            "Đau bụng", "Đau đầu", "Đau lưng", "Đau ở vị trí khác / khó xác định",
        ], r"\b(bung|dau dau|nhuc dau|lung|nguc|hong|chan|tay|vai|rang|dau co)\b"),
        (r"\b(buon non|non|chong mat|dau dau|nhuc dau|met|met moi|sot|phu|sung|ngua|ho|kho chiu|tieu buot|ra dich|ra mau)\b",
         "Triệu chứng xuất hiện như thế nào?", [
             "Mới xuất hiện hôm nay", "Đã kéo dài vài ngày", "Thỉnh thoảng xuất hiện rồi hết", "Tôi chưa xác định được thời điểm",
         ], r"\b(\d+\s*(gio|ngay|tuan|thang)|hom nay|hom qua|tu sang|tu toi|keo dai|lien tuc|thinh thoang|du doi|nhe|nhieu|it|tung con|moi sang)\b"),
        (r"\b(trieu chung|bat thuong|khong khoe|kho o|bi dau)\b", "Bạn muốn tìm hiểu về triệu chứng nào?", [
            "Đau hoặc khó chịu ở bụng", "Buồn nôn hoặc nôn", "Chóng mặt hoặc mệt mỏi", "Triệu chứng khác / chưa xác định",
        ], r"(?!)"),
    ]
    for pattern, title, labels, detail_pattern in groups:
        if re.search(pattern, text) and not re.search(detail_pattern, text):
            return {
                "title": title,
                "description": "Chọn mô tả gần nhất với tình trạng của bạn để làm rõ câu hỏi, hoặc tự bổ sung chi tiết.",
                "options": [{"id": str(i), "label": label} for i, label in enumerate(labels)],
            }
    return None
