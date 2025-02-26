# Intruder and Animal Detection System

This project is an AI-powered **Intruder and Animal Detection System** that uses YOLOv8, YOLOv5, and Face Recognition to detect intruders, identify family members, and detect animals near your property. It also provides real-time alerts via Telegram and triggers automated alarms based on detected threats.

## Features

✅ **Intruder Detection:** Uses YOLOv5 to detect people in the live camera feed.
✅ **Face Recognition:** Identifies family members and flags unknown individuals.
✅ **Animal Detection:** Uses YOLOv8 to detect animals like dogs, cats, and elephants.
✅ **Telegram Alerts:** Sends alerts with detected images to a Telegram bot.
✅ **MongoDB Storage:** Stores detection logs, timestamps, and alert history.
✅ **Alarm System:** Triggers different alarm sounds for different detected animals.
✅ **Multi-threading:** Uses `ThreadPoolExecutor` for improved performance.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/intruder-animal-detection.git
cd intruder-animal-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Telegram Bot
- Create a Telegram bot via [BotFather](https://t.me/BotFather)
- Get the bot token and chat ID
- Update `config.py` with your token and chat ID

### 4. Run the System
```bash
python main.py
```

## Configuration
Modify `config.py` to update:
- Telegram bot credentials
- YOLO model paths
- MongoDB connection details

## Future Enhancements
- 🚀 **Auto Mode for Nighttime Alerts**
- 🎯 **Improved Anomaly Detection**
- 📡 **Integration with IoT Devices**

## Contributing
Feel free to open an issue or submit a pull request to improve the project.

## License
This project is licensed under the MIT License.

