<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

  <h1>Tên dự án: CCAT - Collaborative Computerized Adaptive Testing</h1>

  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#gioi-thieu">Giới thiệu về dự án</a></li>
    <li><a href="#ccat">CCAT - Collaborative Computerized Adaptive Testing</a></li>
    <li><a href="#danh-gia">Đánh giá về thuật toán CCAT</a></li>
    <li><a href="#tich-hop-llm">Tích hợp LLM</a></li>
    <li><a href="#chay-local">Cách để chạy local</a></li>
    <li><a href="#huong-dan">Hướng dẫn người dùng sử dụng ứng dụng</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>

  <hr>

  <h2 id="gioi-thieu">1. Giới thiệu về dự án</h2>
  <p>Dự án CCAT là hệ thống kiểm tra thích ứng giúp đánh giá năng lực của học sinh dựa trên các bài kiểm tra được cá nhân hóa. Hệ thống nhằm tối ưu hóa quá trình đánh giá và cải thiện trải nghiệm học tập.</p>

  <hr>

  <h2 id="ccat">2. CCAT - Collaborative Computerized Adaptive Testing</h2>
  <p>CCAT là một phương pháp kiểm tra hiện đại sử dụng thuật toán tự động điều chỉnh độ khó của câu hỏi dựa trên hiệu suất của học sinh. Hệ thống này giúp tạo ra các bài kiểm tra phù hợp và chính xác hơn trong việc đánh giá năng lực.</p>

  <hr>

  <h2 id="danh-gia">3. Đánh giá về thuật toán CCAT</h2>
  <p>Thuật toán CCAT phân tích độ khó của các câu hỏi và kết quả của các bài kiểm tra trước đó để liên tục hiệu chỉnh bài kiểm tra hiện tại. Việc đánh giá này giúp cá nhân hóa quá trình kiểm tra, đảm bảo rằng mỗi bài kiểm tra phản ánh đúng khả năng của từng học sinh.</p>

  <h3>Results</h3>
  <p>Our model achieves the following performance on:</p>

  <h4>Intra Ranking Consistency (MCMC)</h4>
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th>NIPS2020</th>
        <th>Step 5</th>
        <th>Step 10</th>
        <th>Step 15</th>
        <th>Step 20</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Random</td>
        <td>0.7411/0.7531</td>
        <td>0.8061/0.8084</td>
        <td>0.8348/0.8363</td>
        <td>0.8540/0.8547</td>
      </tr>
      <tr>
        <td><a href="https://api.taylorfrancis.com/content/books/mono/download?identifierName=doi&amp;identifierValue=10.4324/9780203056615&type=googlepdf" target="_blank"><u>FSI</u></a></td>
        <td>0.7912/0.7933</td>
        <td>0.8570/0.8573</td>
        <td>0.8846/0.8848</td>
        <td>0.8975/<strong>0.8977</strong></td>
      </tr>
      <tr>
        <td><a href="https://journals.sagepub.com/doi/abs/10.1177/014662169602000303" target="_blank"><u>KLI</u></a></td>
        <td>0.7821/0.7839</td>
        <td>0.8532/0.8530</td>
        <td>0.8804/0.8805</td>
        <td>0.8965/0.8966</td>
      </tr>
      <tr>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9338437/" target="_blank"><u>MAAT</u></a></td>
        <td>0.6762/0.6909</td>
        <td>0.8083/0.8090</td>
        <td>0.8588/0.8595</td>
        <td>0.8843/0.8848</td>
      </tr>
      <tr>
        <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/20399" target="_blank"><u>NCAT</u></a></td>
        <td>0.7766/0.7923</td>
        <td>0.8451/0.8501</td>
        <td>0.8710/0.8725</td>
        <td>0.8831/0.8840</td>
      </tr>
      <tr>
        <td><a href="https://nips.cc/virtual/2023/poster/70224" target="_blank"><u>BECAT</u></a></td>
        <td>0.7685/0.7680</td>
        <td>0.8441/0.8449</td>
        <td>0.8766/0.8771</td>
        <td>0.8958/0.8961</td>
      </tr>
      <tr>
        <td>CCAT</td>
        <td>0.7982/<strong>0.8149</strong></td>
        <td>0.8561/<strong>0.8635</strong></td>
        <td>0.8832/<strong>0.8851</strong></td>
        <td>0.8955/0.8969</td>
      </tr>
    </tbody>
  </table>
  <p>📋 The results in each grid are evaluated using IRT on the left and CCAT on the right, respectively.</p>

  <h4>Inter Ranking Consistency</h4>
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th>NIPS2020</th>
        <th>Step 5</th>
        <th>Step 10</th>
        <th>Step 15</th>
        <th>Step 20</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Random</td>
        <td>0.7798</td>
        <td>0.8325</td>
        <td>0.8590</td>
        <td>0.8760</td>
      </tr>
      <tr>
        <td><a href="https://api.taylorfrancis.com/content/books/mono/download?identifierName=doi&amp;identifierValue=10.4324/9780203056615&type=googlepdf" target="_blank"><u>FSI</u></a></td>
        <td>0.8258</td>
        <td>0.8785</td>
        <td>0.9013</td>
        <td><strong>0.9126</strong></td>
      </tr>
      <tr>
        <td><a href="https://journals.sagepub.com/doi/abs/10.1177/014662169602000303" target="_blank"><u>KLI</u></a></td>
        <td>0.8195</td>
        <td>0.8758</td>
        <td>0.8985</td>
        <td>0.9119</td>
      </tr>
      <tr>
        <td><a href="https://ieeexplore.ieee.org/abstract/document/9338437/" target="_blank"><u>MAAT</u></a></td>
        <td>0.7242</td>
        <td>0.8373</td>
        <td>0.8807</td>
        <td>0.9023</td>
      </tr>
      <tr>
        <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/20399" target="_blank"><u>NCAT</u></a></td>
        <td>0.8286</td>
        <td>0.8697</td>
        <td>0.8892</td>
        <td>0.8994</td>
      </tr>
      <tr>
        <td><a href="https://nips.cc/virtual/2023/poster/70224" target="_blank"><u>BECAT</u></a></td>
        <td>0.8045</td>
        <td>0.8676</td>
        <td>0.8948</td>
        <td>0.9104</td>
      </tr>
      <tr>
        <td>CCAT</td>
        <td><strong>0.8476</strong></td>
        <td><strong>0.8839</strong></td>
        <td><strong>0.9013</strong></td>
        <td>0.9116</td>
      </tr>
    </tbody>
  </table>

  <hr>

  <h2 id="tich-hop-llm">4. Tích hợp LLM</h2>
  <p>Để nâng cao hiệu quả hỗ trợ người dùng, dự án tích hợp mô hình LLM (Large Language Model). Mô hình này cung cấp các gợi ý và bài tập luyện tập nhằm cải thiện hiệu quả học tập cho học sinh. Qua đó, LLM đóng vai trò trợ giúp trong việc cá nhân hóa đề thi và hỗ trợ tư vấn học tập.</p>

  <hr>

  <h2 id="chay-local">5. Cách để chạy local</h2>
  <p>Dưới đây là hướng dẫn cài đặt và chạy dự án trên máy tính cá nhân:</p>

  <h3>5.1. Chạy giao diện người dùng (UI - React)</h3>
  <pre><code class="language-bash">
npm start
</code></pre>
  <p>Giao diện React sẽ được khởi chạy tại: <a href="http://localhost:5000" target="_blank">http://localhost:5173</a></p>

  <h3>5.2. Chạy Backend chính (CCAT)</h3>
  <pre><code class="language-bash">cd CCCAT
pip install -r requirements.txt
python main.py --reload</code></pre>
  <p>Backend sẽ chạy tại: <a href="http://localhost:5000" target="_blank">http://localhost:8000</a></p>

  <h3>Requirements</h3>
  <p>Tải các thư viện cần thiết trong requirements:</p>
  <pre><code class="language-setup">pip install -r requirements.txt</code></pre>

  <h3>Data Preprocessing</h3>
  <p><strong>Trong Repository này đã preprocess data nên có thể skip bước này.</strong></p>
  <p>Cấu trúc lại thư mục như sau:</p>
  <pre><code class="language-train">
data/
│
├── NIPS2020/
│   ├── train_task_3_4.csv
│   └── meta_data.csv
│
│
└── dataset.py
└── mcmc.py
└── prepare_data.py
└── setting.py
  </code></pre>
  <p>To preprocessing the dataset, run:</p>
  <pre><code class="language-train">cd data
python prepare_data.py --data_name='NIPS2020'</code></pre>
  <p>📋 <strong>prepare_data.py</strong> will delete students with less than 50 answering records, as well as delete questions with less than 50 answering times. The dataset will be divided into a training set (collaborative students) and a testing set (tested students) in a 4:1 ratio. The outputs of prepare_data.py are <strong>train_triples.csv</strong>, <strong>test_triples.csv</strong>, <strong>triples.csv</strong>, <strong>metadata.json</strong>, <strong>concept_map.json</strong>.</p>

  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th>Dataset</th>
        <th>NIPS-EDU(NIPS2020)</th>
        <th>JUNYI</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>#Students</td>
        <td>4,914</td>
        <td>8,852</td>
      </tr>
      <tr>
        <td>#Questions</td>
        <td>900</td>
        <td>702</td>
      </tr>
      <tr>
        <td>#Response logs</td>
        <td>1,382,173</td>
        <td>801,270</td>
      </tr>
      <tr>
        <td>#Response logs per student</td>
        <td>281.27</td>
        <td>90.52</td>
      </tr>
      <tr>
        <td>#Response logs per question</td>
        <td>1,535.75</td>
        <td>1,141.41</td>
      </tr>
    </tbody>
  </table>
  <p>To get the parameter of <font size="4"><a href="https://link.springer.com/book/10.1007/978-0-387-89976-3" target="_blank">IRT</a></font> estimated by mcmc method, run:</p>
  <pre><code class="language-train">python mcmc.py --data_name='NIPS2020'</code></pre>
  <p>📋 <strong>mcmc.py</strong> will use Monte Carlo sampling on the dataset to perform posterior estimation on the IRT model, in order to obtain the parameters of the IRT model. The outputs are <strong>alpha.npy</strong> and <strong>beta.npy</strong>, which contain the discrimination and difficulty of questions.</p>
  <p>After the data preprocess, the folder becomes (which is provided):</p>
  <pre><code class="language-train">
data/
│
├── NIPS2020/
│   ├── alpha.npy
│   ├── beta.npy
│   ├── concept_map.json
│   ├── metadata.json
│   ├── test_triples.csv
│   ├── train_triples.csv
│   └── triples.csv
  </code></pre>

  <hr>
  <h3>5.3. Chạy Backend trong folder <code>LLM</code></h3>
  <pre><code class="language-bash">cd LLM
pip install -r requirements.txt
python app.py</code></pre>
  <p>API LLM sẽ chạy tại: <a href="http://localhost:5000" target="_blank">http://localhost:8500</a> (hoặc port được cấu hình trong file <code>app.py</code>)</p>
 
  <hr>
  <h2>6. Hướng dẫn người dùng sử dụng ứng dụng</h2>
  <ol>
    <li>
      Mở trình duyệt và truy cập: 
      <a href="http://localhost:5173" target="_blank">http://localhost:5173</a>
    </li>
    <li>
      Đăng ký hoặc đăng nhập bằng tài khoản dành cho học sinh hoặc giáo viên.
      <br>
      <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/image/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202025-04-13%20020655.png" alt="Ảnh 1: Giao diện đăng nhập" style="max-width:100%;">
    </li>
    <li>
      <strong>Trải nghiệm khóa học:</strong>
      <ul>
        <li>
          Khi nhấn vào <em>Course</em>, giao diện các khóa học sẽ hiển thị.
          <br>
          <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/image/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202025-04-13%20021027.png" alt="Ảnh 2: Giao diện danh sách khóa học" style="max-width:100%;">
        </li>
        <li>
          Khi chọn một khóa học, bạn sẽ được chuyển đến trang hiển thị video bài giảng và chatbot hỗ trợ.
          <br>
          <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/%E1%BA%A2nh%20video.png" alt="Ảnh 3: Video bài giảng và chatbot hỗ trợ" style="max-width:100%;">
                <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/AI.png" alt="Ảnh 4: Giao diện dashboard" style="max-width:100%;">
        </li>
      </ul>
    </li>
    <li>
      <strong>Dashboard:</strong>
      Khi bấm vào dashboard, giao diện sẽ hiển thị theo như hình bên dưới:
      <br>
      <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202025-04-13%20025605.png" alt="Ảnh 4: Giao diện dashboard" style="max-width:100%;">
      
    </li>
    <li>
      <strong>Bài kiểm tra đánh giá năng lực:</strong>
      Giao diện thực hiện bài test đánh giá trình độ dựa trên năng lực thực sẽ hiển thị như bên dưới. Các câu hỏi hiện đang được lấy từ dataset NIPS 2020, các câu hỏi định dạng trắc nghiệm 4 đáp án đúng. Ngoài ra, website cũng hiển thị thêm hai đồ thị, trong đó đồ thị đường thể hiện sự thay đổi theta ứng với mỗi câu trả lời của học sinh còn biểu đồ phân phối chuẩn thể hiện điểm số của em học sinh đó so với các 'anchor student' đã được hệ thống thu thập trước đó.
      <br>
      <img src="https://github.com/Namtk214/Collaborative-Computerized-Adaptive-Testing/blob/main/b%C3%A0i%20l%C3%A0m.png" alt="Ảnh 5: Giao diện bài test đánh giá năng lực" style="max-width:100%;">
    </li>
  </ol>

  <h2 id="license">7. License</h2>
  <p>Dự án này được cấp phép theo <strong>MIT License</strong>. Vui lòng xem file LICENSE để biết thêm thông tin chi tiết.</p>

  <hr>

  <h2 id="citation">8. Citation</h2>
  <p>Nếu bạn sử dụng hoặc chia sẻ dự án, vui lòng trích dẫn theo thông tin sau:</p>
  <pre><code>
Tác giả: [Tên tác giả hoặc nhóm phát triển]
Dự án: CCAT - Collaborative Computerized Adaptive Testing
Liên kết: [URL dự án hoặc repository]
  </code></pre>

</body>
</html>
