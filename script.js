async function predictWeather() {
    const temperature = document.getElementById('temperature').value;
    const humidity = document.getElementById('humidity').value;
    const windSpeed = document.getElementById('windSpeed').value;
    const pressure = document.getElementById('pressure').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                temperature: parseFloat(temperature),
                humidity: parseFloat(humidity),
                windSpeed: parseFloat(windSpeed),
                pressure: parseFloat(pressure)
            })
        });

        if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
        
        const data = await response.json();
        document.getElementById('weather-result').innerText = data.prediction;
    } catch (error) {
        console.error('Error predicting weather:', error);
        document.getElementById('weather-result').innerText = "Error: Could not retrieve prediction";
    }
}
