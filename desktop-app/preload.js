// preload.js - Securely exposes backend functionality to the frontend UI.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Expose a function that the HTML can call to start the analysis
    runAnalysis: () => ipcRenderer.invoke('run-analysis'),
    // Expose a function to receive real-time log updates from the backend
    onLogUpdate: (callback) => ipcRenderer.on('analysis-log', (_event, value) => callback(value))
});
