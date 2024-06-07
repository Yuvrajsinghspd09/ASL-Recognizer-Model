// Function to capture video frame and send for prediction
async function predictFromVideo() {
    const video = document.getElementById('videoElement');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas image to base64 data
    const imageData = canvas.toDataURL('image/jpeg'); // You may need to adjust this based on image format
    
    // Send image data to server
    const response = await fetch('http://localhost:8000/predict', {  // Modified endpoint
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
    });

    try {
        const data = await response.json();
        console.log('Predicted Letter:', data.predicted_letter);
        // Perform actions with predicted letter (e.g., display on the webpage)
    } catch (error) {
        console.error('Parsing response failed:', error);
        // Handle error if parsing response fails
    }
}

// Add event listener to the predict button
const predictButton = document.getElementById('predictButton');
predictButton.addEventListener('click', predictFromVideo);
