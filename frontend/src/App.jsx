import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle image file input
  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  // Handle form submission to send the image to the server
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      setError('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);  // The key 'image' should match the backend

    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Expecting a class name or a prediction index from backend
      setPrediction(response.data.predictions[0].class);  // This should match the backend output
      setError(null);
    } catch (err) {
      setError('Prediction failed. Please try again.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  console.log(prediction, "class")

  return (
    <div className="App" style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Image Prediction App</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button type="submit" disabled={loading}>Predict</button>
      </form>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default App;
