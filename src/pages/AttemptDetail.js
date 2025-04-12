// AttemptDetail.js
import React from 'react';
import { useParams, Link } from 'react-router-dom';
import './AttemptDetail.css';

export default function AttemptDetail() {
  const { id } = useParams();

  // Dữ liệu mẫu cho attempt id = 1
  // Trong thực tế bạn có thể lấy dữ liệu dựa theo id
  const attemptDetail = {
    id: 1,
    attemptLabel: "Lần thử 1",
    candidateName: "Nguyễn Văn A",
    date: "2023-03-15",
    time: "14:00",
    score: "15/20",
    // Giả sử 15 câu đúng, 5 câu sai (theo thứ tự)
    questions: Array.from({ length: 20 }, (_, index) => ({
      id: index + 1,
      correct: index < 15,
    }))
  };

  // Nếu id khác 1, có thể hiển thị dữ liệu tương ứng hoặc thông báo không tìm thấy

  return (
    <div className="attempt-container">
      <div className="attempt-header">
        <h2>{attemptDetail.attemptLabel}</h2>
        <p>Chi tiết thông tin thí sinh và kết quả 20 câu hỏi</p>
      </div>

      <div className="candidate-info">
        <h3>Thông tin thí sinh</h3>
        <p><strong>Tên:</strong> {attemptDetail.candidateName}</p>
        <p><strong>Ngày:</strong> {attemptDetail.date}</p>
        <p><strong>Thời gian:</strong> {attemptDetail.time}</p>
        <p><strong>Điểm số:</strong> {attemptDetail.score}</p>
      </div>

      <div>
        <h3>Kết quả 20 câu hỏi</h3>
        <div className="question-list">
          {attemptDetail.questions.map((question) => (
            <div 
              key={question.id} 
              className={`question-card ${question.correct ? 'correct' : 'wrong'}`}
            >
              <h4>Câu {question.id}</h4>
              <p>{question.correct ? "Đúng" : "Sai"}</p>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: '24px', textAlign: 'center' }}>
        <Link to="/history" className="back-link">← Quay lại History</Link>
      </div>
    </div>
  );
}
