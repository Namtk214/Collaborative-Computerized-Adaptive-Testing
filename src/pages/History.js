// History.js
import React from 'react';
import { Link } from 'react-router-dom';
import './History.css';

export default function History() {
  const historyData = [
    { id: 1, "Lần thử": "Lần thử 1", date: "2023-03-15", time: "14:00", "điểm": "15/20" },
    { id: 2, "Lần thử": "Lần thử 2", date: "2023-03-16", time: "09:30", "điểm": "12/20" },
    { id: 3, "Lần thử": "Lần thử 3", date: "2023-03-18", time: "20:45", "điểm": "18/20" },
    { id: 4, "Lần thử": "Lần thử 4", date: "2023-03-20", time: "18:10", "điểm": "10/20" },
    { id: 5, "Lần thử": "Lần thử 5", date: "2023-03-22", time: "11:15", "điểm": "16/20" },
    { id: 6, "Lần thử": "Lần thử 6", date: "2023-03-24", time: "16:30", "điểm": "14/20" },
    { id: 7, "Lần thử": "Lần thử 7", date: "2023-03-25", time: "10:20", "điểm": "17/20" },
    { id: 8, "Lần thử": "Lần thử 8", date: "2023-03-27", time: "08:50", "điểm": "11/20" },
    { id: 9, "Lần thử": "Lần thử 9", date: "2023-03-28", time: "13:05", "điểm": "19/20" },
    { id: 10, "Lần thử": "Lần thử 10", date: "2023-03-29", time: "15:40", "điểm": "13/20" },
    { id: 11, "Lần thử": "Lần thử 11", date: "2023-03-30", time: "17:20", "điểm": "12/20" },
    { id: 12, "Lần thử": "Lần thử 12", date: "2023-03-31", time: "12:00", "điểm": "14/20" },
    { id: 13, "Lần thử": "Lần thử 13", date: "2023-04-01", time: "09:00", "điểm": "16/20" },
    { id: 14, "Lần thử": "Lần thử 14", date: "2023-04-02", time: "19:00", "điểm": "10/20" },
    { id: 15, "Lần thử": "Lần thử 15", date: "2023-04-03", time: "14:30", "điểm": "18/20" },
    { id: 16, "Lần thử": "Lần thử 16", date: "2023-04-04", time: "11:45", "điểm": "15/20" },
  ];

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>History</h2>
        <p>Xem lại danh sách lịch sử hoạt động của bạn</p>
      </div>

      <div className="history-list">
        {historyData.map((item) => (
          <Link
            to={`/attempt/${item.id}`}
            key={item.id}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            <div className="history-card">
              <h4>{item["Lần thử"]}</h4>
              <div className="history-info">
                <span>Ngày: {item.date}</span>
                <span>Thời gian: {item.time}</span>
                <span>Điểm: {item["điểm"]}</span>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
