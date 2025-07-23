
import React, { useState } from 'react';

function App() {
  const [amount, setAmount] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch('https://your-backend.onrender.com/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount, location: 'Dubai', time: 'now' })
    });
    const data = await res.json();
    setResult(data.prediction);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Fraud Detector</h1>
      <form onSubmit={handleSubmit}>
        <input value={amount} onChange={(e) => setAmount(e.target.value)} placeholder="Enter amount" />
        <button type="submit">Check</button>
      </form>
      {result && <p>Result: {result}</p>}
      <iframe src="https://huggingface.co/chat" width="100%" height="400px" frameBorder="0" title="Chatbot" />
    </div>
  );
}

export default App;
