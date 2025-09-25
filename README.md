# Machine_translation


# Hệ thống Dịch máy Anh-Việt sử dụng Kiến trúc Transformer

Dự án này trình bày việc xây dựng và tối ưu hóa một hệ thống dịch máy thần kinh (Neural Machine Translation - NMT) cho cặp ngôn ngữ Anh-Việt dựa trên kiến trúc Transformer.

[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)

-----

## 📜 Giới thiệu

Hệ thống này được xây dựng dựa trên mô hình Transformer, một kiến trúc tiên tiến trong lĩnh vực xử lý ngôn ngữ tự nhiên, để dịch từ tiếng Anh sang tiếng Việt. Dự án không chỉ tập trung vào việc đạt được chất lượng dịch thuật cao mà còn chú trọng đến các kỹ thuật tối ưu hóa mô hình để có thể triển khai trong các ứng dụng thực tế.

Các thành phần chính của dự án bao gồm:

  * **Kiến trúc Transformer**: Tận dụng cơ chế tự chú ý (self-attention) để xử lý các phụ thuộc xa trong câu.
  * **Tokenization**: Sử dụng SentencePiece để tạo ra các token ở mức subword, giúp xử lý từ vựng lớn và các từ không có trong từ điển.
  * **Tối ưu hóa**: Áp dụng các kỹ thuật như lượng tử hóa động (dynamic quantization) và xuất mô hình sang định dạng ONNX để giảm kích thước và tăng tính tương thích.

-----

## 🚀 Tính năng nổi bật

  * **Chất lượng dịch thuật cạnh tranh**: Đạt điểm BLEU **24.30** với chiến lược giải mã Beam Search.
  * **Mô hình hiệu quả**: Giảm kích thước mô hình khoảng **74.64%** (từ 229.75 MB xuống 58.25 MB) thông qua lượng tử hóa động, giúp triển khai dễ dàng hơn trên các thiết bị có tài nguyên hạn chế.
  * **Tính di động**: Hỗ trợ xuất mô hình sang định dạng ONNX, tạo điều kiện thuận lợi cho việc triển khai trên nhiều nền tảng khác nhau.

-----

## 🛠️ Kiến trúc và Phương pháp

### **1. Dữ liệu và Tiền xử lý**

  * **Tập dữ liệu**: Sử dụng bộ dữ liệu song ngữ Anh-Việt IWSLT15 từ Hugging Face Datasets.
  * **Tiền xử lý**: Dữ liệu được làm sạch bằng cách loại bỏ khoảng trắng thừa, chuyển thành chữ thường và chuẩn hóa để đảm bảo tính nhất quán.

### **2. Tokenization**

  * Chúng tôi sử dụng **SentencePiece** để huấn luyện các tokenizer riêng biệt cho tiếng Anh và tiếng Việt.
  * Kích thước từ vựng là **30,000** cho tiếng Anh và **15,000** cho tiếng Việt.

### **3. Kiến trúc Mô hình**

  * Mô hình được xây dựng dựa trên kiến trúc Transformer tiêu chuẩn, bao gồm một bộ mã hóa (encoder) và một bộ giải mã (decoder).
  * **Hyperparameters**:
      * Số lớp Encoder & Decoder: 4
      * Số lượng attention heads: 4
      * Kích thước embedding: 512
      * Tỷ lệ Dropout: 0.25

### **4. Huấn luyện**

  * **Trình tối ưu hóa**: AdamW.
  * **Hàm mất mát**: Cross-Entropy với kỹ thuật làm mịn nhãn (label smoothing).
  * **Bộ điều chỉnh tốc độ học**: Cosine Annealing.
  * Mô hình được huấn luyện trong 30 epochs với cơ chế dừng sớm (early stopping) để tránh overfitting.

-----

## 📊 Kết quả

### **Chất lượng dịch thuật**

Hệ thống được đánh giá bằng điểm **BLEU** trên tập validation.

| Chiến lược giải mã | Điểm BLEU |
| :--- | :---: |
| Greedy Decoding | 23.53 |
| Beam Search (Beam Width = 4) | **24.30** |

Kết quả cho thấy Beam Search mang lại hiệu suất dịch tốt hơn so với Greedy Decoding.

### **Hiệu quả Mô hình**

Kỹ thuật lượng tử hóa động đã giảm đáng kể kích thước mô hình mà vẫn duy trì chất lượng dịch thuật.

  * **Kích thước mô hình gốc**: 229.75 MB
  * **Kích thước mô hình đã lượng tử hóa**: 58.25 MB (Giảm **74.64%**)

-----

## ⚙️ Hướng dẫn cài đặt và Sử dụng

### **Yêu cầu**

  * Python 3.8+
  * PyTorch
  * Hugging Face Datasets
  * SentencePiece
  * (Các thư viện khác được liệt kê trong `requirements.txt`)

### **Các bước cài đặt**

1.  **Clone repository:**

    ```bash
    git clone https://github.com/VMSSON345/Machine_translation.git
    ``

2.  **Cài đặt các gói phụ thuộc:**

    ```bash
    pip install -r requirements.txt
    ```

### **Sử dụng mô hình để dịch**

Bạn có thể sử dụng script được cung cấp để dịch một câu từ tiếng Anh sang tiếng Việt.

```bash
python translate.py --text "Glad to see you here!" --model_path "path/to/your/best_model.pt"
```

-----

## 💡 Hướng phát triển trong tương lai

  * **Tăng cường dữ liệu**: Áp dụng các kỹ thuật như back-translation để cải thiện độ mạnh mẽ của mô hình.
  * **Tinh chỉnh trên các tập dữ liệu chuyên ngành**: Tùy chỉnh mô hình cho các lĩnh vực cụ thể như y tế hoặc công nghệ để tăng độ chính xác.
  * **Đánh giá hiệu suất**: Đo lường độ trễ và thông lượng của mô hình đã lượng tử hóa trên các thiết bị phần cứng khác nhau.

-----

## 🙏 Lời cảm ơn

Dự án này được xây dựng dựa trên các nghiên cứu và công cụ mã nguồn mở tuyệt vời từ cộng đồng. Chúng tôi xin chân thành cảm ơn các tác giả của những công trình đã được trích dẫn.

-----

## 📄 Giấy phép

Dự án này được cấp phép theo Giấy phép MIT. Xem chi tiết tại file `LICENSE`.
