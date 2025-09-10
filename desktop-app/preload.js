// preload.js - Securely exposes backend functionality to the frontend UI.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Expose a function that takes the file path and sends it to the backend
    runAnalysis: (filePath) => ipcRenderer.invoke('run-analysis', filePath),
    // Expose a function to receive real-time log updates
    onLogUpdate: (callback) => ipcRenderer.on('analysis-log', (_event, value) => callback(value))
});

