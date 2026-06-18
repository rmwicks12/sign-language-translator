# Mudrā: Real-Time Sign Language Translation System

## Overview
Mudrā is a high-throughput, low-latency continuous sign language translation system designed to bridge the communication gap through real-time text and audio feedback. Engineered by Rithwik Girish Murthy, the system decouples frontend spatial feature extraction from backend deep learning inference, establishing a seamless 30-frame sliding window matrix ingestion loop without frame drops or system bottlenecks.

## How It Works
The architectural pipeline follows a highly optimized, asynchronous data flow:

1. **Data Capture:** A standard web camera captures user gestures.
2. **Frontend Feature Extraction:** The client-side browser utilizes MediaPipe to extract 21 discrete 3D skeletal joint landmarks per hand. These points are mathematically unrolled into a flat, 126-feature data array.
3. **Asynchronous Transmission:** The spatial coordinates are streamed via persistent WebSockets to the backend consumer loop every 150 milliseconds.
4. **Sequential Inference:** A deep Long Short-Term Memory (LSTM) recurrent neural network evaluates the time-series coordinates over a rolling 30-frame sliding window, rather than evaluating isolated static images.
5. **Database Logging:** Verified translations (confidence > 0.75) are securely logged to a relational database using thread-safe wrappers.
6. **Real-Time UI Output:** The predicted classification string is routed back to the frontend dashboard. The interface updates the continuous typewriter sentence tray and leverages browser APIs to trigger automated text-to-speech audio rendering.

## Technology Stack & Plugins

### Backend & Networking
* **Python 3.10+:** The core runtime environment driving the backend logic, data manipulation, and machine learning wrappers.
* **Django:** The primary web framework structuring the application backend.
* **Django Channels & Daphne (ASGI):** Replaces traditional, blocking synchronous HTTP requests with an Asynchronous Server Gateway Interface (ASGI). This powers the bidirectional WebSocket pipelines, allowing continuous 150ms matrix streaming without connection handshake overhead.

### Machine Learning Core
* **TensorFlow & Keras:** Powers the core deep Long Short-Term Memory (LSTM) neural network (`mudra_lstm_model.h5`). The LSTM structure is critical for this project because it retains memory of sequential movements across the 30-frame window, achieving a 97.5% categorical accuracy.
* **NumPy:** Handles high-speed, multi-dimensional array manipulation on the backend. It actively manages the sliding window queues and executes spatial relative-wrist coordinate normalization prior to model inference.

### Frontend & Data Extraction
* **MediaPipe Hands:** Initialized via JavaScript to track dual-hand placements in real-time. Moving this intensive feature extraction load to the client-side prevents the server from suffering pixel-processing bottlenecks.
* **Web Speech API:** A native browser API that provides instant text-to-speech audio rendering when a user triggers the voice broadcast option on the compiled text tray.
* **HTML5, CSS3, & Bootstrap:** Structures the responsive administration dashboard, embedding live camera canvas overlays and predictive confidence gauges.

### Data Persistence & MLOps
* **PostgreSQL:** A highly robust relational database system utilized to safely record timestamped translation strings, session metrics, and historical logging logs directly from the asynchronous backend threads.
* **Custom MLOps Engine:** A hot-swapping Python monitoring thread that tracks model weight modification timestamps on the disk. This allows the system to reload updated `.h5` neural networks into live RAM seamlessly, without requiring a server restart or dropping active user connections.
