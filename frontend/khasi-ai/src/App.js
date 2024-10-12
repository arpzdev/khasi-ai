import React, { useState, useRef, useEffect } from 'react'; 
import './App.css'; // Add custom CSS styles here
import axios from 'axios';
import companyLogo from './assets/companyLogo.png';
import khasiLogo from './assets/khasiLogo.png';
import uploadIcon from './assets/UploadIcon.png';
import cameraIcon from './assets/CameraIcon.png';
import fbIcon from './assets/FbIcon.png';
import instaIcon from './assets/InstaIcon.png';
import xIcon from './assets/XIcon.png';
import linkedinIcon from './assets/LinkedInIcon.png';
import ytIcon from './assets/YTIcon.png';
import tiktokIcon from './assets/tiktokIcon.png';

function App() {
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [errorMessage, setErrorMessage] = useState(''); // State for error messages
    const [usingCamera, setUsingCamera] = useState(false);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null); // Store the media stream

    // Function to handle file upload
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImage(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    // Function to upload image
    const uploadImage = async () => {
        const formData = new FormData();
        const blob = await fetch(image).then((res) => res.blob());
        formData.append('file', blob, 'uploaded_image.jpg');

        try {
            const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            
            // Handle predictions and error message
            if (response.data.prediction === 'meat image not recognized') {
                setErrorMessage('Meat image not recognized.');
                setPrediction(''); // Clear previous prediction
            } else {
                setPrediction(response.data.prediction);
                setErrorMessage(''); // Clear error message if a valid prediction is made
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            setErrorMessage('Error uploading image');
            setPrediction(''); // Clear prediction if there's an error
        }
    };

    // Function to start camera
    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            setStream(mediaStream); // Save the media stream
            setUsingCamera(true); // Update the state to use the camera
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    };

    // Function to capture image from the camera
    const captureImage = () => {
        const canvas = canvasRef.current;
        if (canvas && videoRef.current) {
            const context = canvas.getContext('2d');
            canvas.width = videoRef.current.videoWidth;  // Set canvas width to video width
            canvas.height = videoRef.current.videoHeight; // Set canvas height to video height
            context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL();
            setImage(dataUrl);
            stopCamera();
        } else {
            console.error("Canvas or video reference is null.");
        }
    };

    // Function to stop camera
    const stopCamera = () => {
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach((track) => track.stop());
        }
        setUsingCamera(false);
        setStream(null); // Clear the stream reference
    };

    // useEffect to assign the stream to the video element
    useEffect(() => {
        if (stream && videoRef.current) {
            videoRef.current.srcObject = stream; // Assign stream to video element
            videoRef.current.play();
        }

        return () => {
            // Clean up the stream when component unmounts or camera stops
            if (stream) {
                stopCamera();
            }
        };
    }, [stream]); // Trigger the effect when the stream changes

    return (
        <div className="App">
            <header className="navbar">
                <img src={companyLogo} alt="Company Logo" className="logo" />
            </header>
            <main className="analyzer-section">
                <img src={khasiLogo} alt="Khasi Logo" className="khasi-logo" />
                <div className="upload-options">
                    <button className="upload-btn" onClick={() => document.getElementById('fileInput').click()}>
                        <img src={uploadIcon} alt="Upload" />
                        <span>Upload</span>
                    </button>
                    <input
                        id="fileInput"
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                    />
                    <button className="camera-btn" onClick={startCamera}>
                        <img src={cameraIcon} alt="Camera" />
                        <span>Camera</span>
                    </button>
                </div>
                {usingCamera ? (
                    <div>
                        <video ref={videoRef} width="300" height="225" />
                        <canvas ref={canvasRef} style={{ display: 'none' }} />
                        <button onClick={captureImage} className="capture-btn">Capture Image</button>
                    </div>
                ) : (
                    image && <img src={image} alt="Preview" width="300" height="225" />
                )}
                {image && <button className="predict-btn" onClick={uploadImage}>Analyze मासु !  </button>}

                {prediction && (
                    <h2 className="prediction-text">
                        {prediction === 'Fresh' && "Freshhh ! मासु  "}
                        {prediction === 'Half-Fresh' && "Half Fresh BUT Rishq हे To Ishq Hein !"}
                        {prediction === 'Spoiled' && "पुरै Spoiled | Love in the time of हैजा."}
                    </h2>
                )}
                
                {errorMessage && <h2 className="error-text">{errorMessage}</h2>}
            </main>
            <footer className="footer">
                <div className="social-links">
                    <a target="_blank" href="https://www.instagram.com/youneedthis.ai/"><img src={instaIcon} alt="Instagram" /></a>
                    <a target="_blank" href="https://www.tiktok.com/@youneedthis_ai"><img src={tiktokIcon} alt="TikTok" /></a>
                    <a target="_blank" href="https://www.facebook.com/profile.php?id=61566759121129"><img src={fbIcon} alt="Facebook" /></a>
                    <a target="_blank" href="https://x.com/youneedthisai"><img src={xIcon} alt="X" /></a>
                    <a target="_blank" href="https://www.linkedin.com/in/you-need-this-ai-b0989332a/"><img src={linkedinIcon} alt="LinkedIn" /></a>
                    <a target="_blank" href="https://www.youtube.com/@youneedthisai/shorts"><img src={ytIcon} alt="YouTube" /></a>
                </div>
            </footer>
        </div>
    );
}

export default App;
