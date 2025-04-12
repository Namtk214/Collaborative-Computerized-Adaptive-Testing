// src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

function Header() {
  return (
    <header className="header-container">
      <div className="header-left">
        <Link to="/" className="logo">
          Computerized Collaborative Adaptive Testing
        </Link>
      </div>
      <div className="header-right">
        <Link to="/test-demo" className="nav-button">Test Demo</Link>
        <Link to="/ai" className="nav-button">AI</Link>
        {/* <Link to="/login" className="login-btn">
          <img
            src="https://img.lovepik.com/png/20231121/the-letter-n-in-a-light-turquoise-circle-vector-ui_659875_wh1200.png"
            alt="Google Logo"
            className="google-logo"
          />
          Đăng nhập
        </Link> */}
      </div>
    </header>
  );
}

export default Header;
