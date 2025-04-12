import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import AIPage from './pages/AIPage';
import TestDemo from './pages/TestDemo';
import History from './pages/History';
import AttemptDetail from './pages/AttemptDetail';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        {/* Bao bọc nội dung chính bằng container theo CSS */}
        <main className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/ai" element={<AIPage />} />
            <Route path="/test-demo" element={<TestDemo />} />
            <Route path="/history" element={<History />} />
            <Route path="/attempt/:id" element={<AttemptDetail />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
