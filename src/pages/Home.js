// Home.js
import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

export default function Home() {
  const daysUntilCCAT = 18;

  return (
    <div className="dashboard-container">
      {/* Sidebar bên trái */}
      <aside className="sidebar">
        <div className="sidebar-logo">CCAT</div>
        <nav className="sidebar-nav">
          <Link to="/ai" className="sidebar-link">AI Chatbot</Link>
          <Link to="/test-demo" className="sidebar-link">Test Demo</Link>
          {/* Thêm link History */}
          <Link to="/history" className="sidebar-link">Lịch sử</Link>
        </nav>
      </aside>

      {/* Main content */}
      <main className="main-content">
        <header className="dashboard-header">
          <h1>Hello Nam 👋</h1>
          <p>It’s <strong>{daysUntilCCAT} days</strong> until your CCAT</p>
        </header>

        <div className="cards-container">
          {/* Hàng thống kê */}
          <section className="stats-grid">
            <div className="card stats-card">
              <h4>All time</h4>
              <div className="stats-row">
                <div>
                  <strong>58%</strong>
                  <small> Accuracy</small>
                </div>
                <div>
                  <strong>3h 36m</strong>
                  <small> Time studied</small>
                </div>
                <div>
                  <strong>64</strong>
                  <small> Questions finished</small>
                </div>
              </div>
            </div>

            <div className="card streak-card">
              <h4>Today's Streak</h4>
              <span className="streak-count">1</span>
              <small>Longest streak: 1</small>
            </div>
          </section>

          {/* Hàng chi tiết */}
          <section className="details-grid">
            <div className="card announcement-card">
              <h5>Announcement Board</h5>
              <ul>
                <li>
                  <strong>Founder’s message:</strong> Chào mừng bạn đến với CCAT
                </li>
                <li>
                  <strong>Introduction:</strong>
                  Collaborative CAT (CCAT) giải quyết vấn đề này bằng cách áp dụng phương pháp học tập hợp tác, cải thiện tính nhất quán trong xếp hạng và chứng minh hiệu quả thực tế trên dữ liệu trong thế giới thực.
                </li>
              </ul>
            </div>

            <div className="card skills-card">
              <h5>Skills</h5>
              <ul>
                <li>Rhetorical Synthesis — Correct 33/55 (Score 60)</li>
                <li>Text Structure & Purpose — Correct 4/9 (Score 44)</li>
                <li>Boundaries — Correct 0/0 (Score 0)</li>
              </ul>
            </div>

            <div className="card recent-card">
              <h5>Recent</h5>
              <ul>
                <li>
                  Rhetorical Synthesis, Medium — 57s — <span className="correct">Correct</span>
                </li>
                <li>
                  Rhetorical Synthesis, Medium — 2m 37s — <span className="correct">Correct</span>
                </li>
              </ul>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
