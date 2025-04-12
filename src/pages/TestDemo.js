import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import NormalDistributionChart from '../components/NormalDistributionChart';
import './TestDemo.css';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

// Custom hook for managing test session
const useTestSession = () => {
  const [testState, setTestState] = useState({
    isStarted: false,
    isCompleted: false,
    question: null,
    result: null,
    selectedAnswer: '',
    error: null,
    lastCorrectness: null
  });

  const startTest = useCallback(async () => {
    try {
      await axios.get(`${API_BASE_URL}/start`, { 
        withCredentials: true 
      });
      fetchNextQuestion();
    } catch (error) {
      setTestState(prev => ({ 
        ...prev, 
        error: 'Failed to start test' 
      }));
    }
  }, []);

  const fetchNextQuestion = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/question`, { 
        withCredentials: true 
      });
      const data = response.data;

      // Check if test is completed
      if (data.result) {
        setTestState(prev => ({
          ...prev,
          isCompleted: true,
          result: {
            score: data.score,
            total: data.total,
            finalTheta: data.final_theta,
            rank: data.current_rank,
            totalAnchor: data.total_anchor
          },
          lastCorrectness: null
        }));
      } else {
        // Get next question
        setTestState(prev => ({
          ...prev,
          isStarted: true,
          question: {
            index: data.current_index + 1,
            total: data.total,
            theta: data.current_theta,
            delta: data.current_delta,
            rank: data.current_rank,
            totalAnchor: data.total_anchor,
            imageUrl: `${API_BASE_URL}${data.image_url}`,
            subjects: data.subjects || [],
            qid: data.qid
          },
          selectedAnswer: '',
          error: null,
          lastCorrectness: null
        }));
      }
    } catch (error) {
      setTestState(prev => ({ 
        ...prev, 
        error: 'Failed to fetch question' 
      }));
    }
  }, []);

  const submitAnswer = useCallback(async (answer) => {
    if (!answer) {
      setTestState(prev => ({ 
        ...prev, 
        error: 'Please select an answer' 
      }));
      return;
    }

    try {
      const formData = new FormData();
      formData.append('answer', answer);

      const response = await axios.post(`${API_BASE_URL}/submit`, formData, { 
        withCredentials: true,
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Update last answer correctness
      setTestState(prev => ({
        ...prev,
        lastCorrectness: response.data.is_correct === 1
      }));

      fetchNextQuestion();
    } catch (error) {
      setTestState(prev => ({ 
        ...prev, 
        error: 'Failed to submit answer' 
      }));
    }
  }, [fetchNextQuestion]);

  return { 
    testState, 
    actions: { 
      startTest, 
      fetchNextQuestion, 
      submitAnswer,
      setSelectedAnswer: (answer) => setTestState(prev => ({ 
        ...prev, 
        selectedAnswer: answer 
      }))
    }
  };
};

export default function TestDemo() {
  const { testState, actions } = useTestSession();
  const { 
    isStarted, 
    isCompleted, 
    question, 
    result, 
    selectedAnswer, 
    error,
    lastCorrectness
  } = testState;

  // ===== Trang đăng nhập (chưa bắt đầu test) =====
  if (!isStarted && !isCompleted) {
    return (
      <div className="test-start">
        <h2>Adaptive Assessment</h2>
        <div className="start-instructions">
          <h3>Test Instructions:</h3>
          <ul>
            <li>20 questions will test your knowledge</li>
            <li>Questions adapt to your performance</li>
            <li>Select the best answer for each question</li>
            <li>Your ability (θ) will be tracked in real-time</li>
          </ul>
        </div>
        {error && <div className="error-message">{error}</div>}
        <button onClick={actions.startTest} className="start-test-btn">Start Test</button>
      </div>
    );
  }

  // ===== Trang kết quả test =====
  if (isCompleted && result) {
    return (
      <div className="test-result">
        <h2>Test Completed</h2>
        <div className="result-details">
          <p>Score: {result.score} / {result.total}</p>
          <p>Ability (θ): {result.finalTheta.toFixed(2)}</p>
          <p>Rank: {result.rank} / {result.totalAnchor}</p>
        </div>
        <div className="result-charts">
          <NormalDistributionChart theta={result.finalTheta} />
        </div>
        <button onClick={actions.startTest}>Retake Test</button>
      </div>
    );
  }

  // ===== Trang làm bài test =====
  return (
    <div className="test-container">
      {question && (
        <div className="test-wrapper">
          {/* Left panel: 2 charts stacked vertically */}
          <div className="charts-panel">
            <img
              src={`${API_BASE_URL}/delta_plot.png?t=${new Date().getTime()}`}
              alt="Performance Deviation"
              className="delta-chart"
            />
            <NormalDistributionChart theta={question.theta} />
          </div>

          {/* Right panel: Question, image, answers, etc. */}
          <div className="question-panel">
            <div className="question-header">
              <h3>Question {question.index} of {question.total}</h3>
              {/* Hiển thị θ và Rank trong container chứa 2 nút */}
              <div className="test-stats">
                <button className="stat-button">
                  θ = {question.theta.toFixed(2)}
                </button>
                <button className="stat-button">
                  Rank: {question.rank} / {question.totalAnchor}
                </button>
              </div>
            </div>

            {lastCorrectness !== null && (
              <div className={`feedback ${lastCorrectness ? 'correct' : 'incorrect'}`}>
                {lastCorrectness ? 'Correct!' : 'Incorrect'}
              </div>
            )}

            {question.imageUrl && (
              <img 
                src={question.imageUrl} 
                alt="Question" 
                className="question-image"
              />
            )}

            {question.subjects && question.subjects.length > 0 && (
              <div className="question-subjects">
                <strong>Subjects:</strong> {question.subjects.join(', ')}
              </div>
            )}

            <div className="answer-options">
              {['A', 'B', 'C', 'D'].map(opt => (
                <button
                  key={opt}
                  className={`answer-btn ${selectedAnswer === opt ? 'selected' : ''}`}
                  onClick={() => actions.setSelectedAnswer(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>

            <button 
              className="submit-btn" 
              disabled={!selectedAnswer}
              onClick={() => actions.submitAnswer(selectedAnswer)}
            >
              Submit Answer
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>
        </div>
      )}
    </div>
  );
}
