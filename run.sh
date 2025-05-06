uvicorn backend.server_async:app --host 0.0.0.0 --port 8000 --ssl-certfile ./backend/localhost+2.pem --ssl-keyfile ./backend/localhost+2-key.pem &
cd frontend/fastrtc-demo && node server.js &
wait