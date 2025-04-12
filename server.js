// server.js
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Route cho trang Test Demo
app.post('/test-demo', (req, res) => {
  const { answer } = req.body;
  // Xử lý logic chấm điểm / demo
  // Ở đây trả về cố định
  return res.json({ message: `Bạn đã nộp: ${answer}. Demo thành công!` });
});

// Route cho trang AI
app.post('/ai', (req, res) => {
  const { prompt } = req.body;
  // Xử lý logic AI / demo
  // Ở đây trả về cố định
  return res.json({ message: `Trả lời cho prompt: "${prompt}". Demo AI thành công!` });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
