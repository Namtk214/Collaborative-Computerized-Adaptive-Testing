import React, { useState, useEffect, useRef } from 'react';
import './AIPage.css';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Send, Image, X, Plus } from 'lucide-react';

function AIPage() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const chatWindowRef = useRef(null);
  const fileInputRef = useRef(null);

  // Format messages with ReactMarkdown
  useEffect(() => {
    setMessages(prev => prev.map(msg => {
      if (msg.role === 'assistant' && !msg.formatted && typeof msg.content === 'string') {
        return { 
          ...msg, 
          content: <ReactMarkdown 
            children={msg.content} 
            remarkPlugins={[remarkMath]} 
            rehypePlugins={[rehypeKatex]} 
          />, 
          formatted: true 
        };
      }
      return msg;
    }));
  }, [messages]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result.startsWith('data:') 
          ? reader.result 
          : `data:image/${file.type.split('/')[1]};base64,${reader.result.split(',')[1]}`;
        
        setSelectedImage(base64String);
        setMessages(prev => [
          ...prev, 
          { 
            role: 'user', 
            content: (
              <div className="image-preview">
                <img 
                  src={base64String} 
                  alt="Uploaded" 
                  className="uploaded-image"
                />
              </div>
            ),
            isImage: true
          }
        ]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAskAI = async () => {
    if (!prompt.trim() && !selectedImage) {
      alert('Please enter a prompt or upload an image');
      return;
    }

    const userMessage = { 
      role: 'user', 
      content: prompt,
      image: selectedImage
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const requestPayload = {
        prompt: prompt || '',
        image: selectedImage || null
      };

      const response = await fetch('http://localhost:5000/ai', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestPayload),
        signal: AbortSignal.timeout(60000)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: data.message || 'No response from AI',
          sources: data.sources || []
        }
      ]);
    } catch (err) {
      console.error("API Error:", err);
      setMessages(prev => [
        ...prev,
        { 
          role: 'assistant', 
          content: `Connection error: ${err.message || 'Unknown error occurred'}`
        }
      ]);
    } finally {
      setIsLoading(false);
      setPrompt('');
      setSelectedImage(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskAI();
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="deepseek-container">
      <div className="deepseek-sidebar">
        <button className="new-chat-btn">
          <Plus className="btn-icon" /> New chat
        </button>
        <div className="chat-history">
          <div className="history-section">Recent Chats</div>
          <div className="history-item">Mathematics Assistant</div>
          <div className="history-item">Science Queries</div>
        </div>
      </div>
      <div className="deepseek-main">
        <div className="chat-header">
          <h2>AI Assistant</h2>
        </div>
        <div className="chat-window" ref={chatWindowRef}>
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              {msg.content}
              {msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  Sources: {msg.sources.join(', ')}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="chat-message assistant loading">
              Thinking...
            </div>
          )}
        </div>
        <div className="chat-input-container">
          {selectedImage && (
            <div className="image-preview-container">
              <img 
                src={selectedImage} 
                alt="Selected" 
                className="selected-image-preview" 
              />
              <button className="remove-image-btn" onClick={clearImage}>
                <X size={16} />
              </button>
            </div>
          )}
          <div className="chat-input">
            <textarea
              rows="2"
              placeholder="Ask something..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyPress={handleKeyPress}
            ></textarea>
            <div className="input-actions">
              <input 
                type="file" 
                ref={fileInputRef}
                accept="image/*" 
                onChange={handleImageUpload} 
                style={{ display: 'none' }}
                id="image-upload"
              />
              <button 
                className="upload-image-btn" 
                onClick={() => fileInputRef.current.click()}
              >
                <Image size={24} />
              </button>
              <button 
                onClick={handleAskAI} 
                disabled={isLoading}
                className="send-btn"
              >
                <Send size={24} />
              </button>
            </div>
          </div>
          <div className="input-footer">
            <span>Powered by AI</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIPage;