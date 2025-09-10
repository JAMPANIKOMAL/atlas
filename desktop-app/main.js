// main.js - This is the "backend" of the desktop application.
// It handles window creation and communication with our Python scripts.

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Function to create the main application window
const createWindow = () => {
    const mainWindow = new BrowserWindow({
        width: 1000,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
        title: "ATLAS"
    });

    mainWindow.loadFile('atlas_app.html');
};

app.whenReady().then(() => {
    createWindow();
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

// This is the core logic that connects our UI to the Python backend.
// It now accepts the file path directly from the UI.
ipcMain.handle('run-analysis', async (event, inputFile) => {
    const isPackaged = app.isPackaged;
    // When packaged, the 'extraResources' are in a predictable location.
    // When in development, they are in the parent directory.
    const projectRoot = isPackaged ? path.join(process.resourcesPath, '..') : path.join(__dirname, '..');
    
    const predictScript = path.join(projectRoot, 'src', 'predict.py');

    // Run the `predict.py` script as a child process
    return new Promise((resolve) => {
        const pythonProcess = spawn('python', [predictScript, '--input_fasta', inputFile], {
            cwd: projectRoot 
        });
        
        pythonProcess.stdout.on('data', (data) => {
            event.sender.send('analysis-log', data.toString());
        });

        pythonProcess.stderr.on('data', (data) => {
            event.sender.send('analysis-log', `ERROR: ${data.toString()}`);
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                const reportFileName = `ATLAS_REPORT_${path.basename(inputFile, path.extname(inputFile))}.txt`;
                const reportPath = path.join(projectRoot, 'reports', reportFileName);
                
                fs.readFile(reportPath, 'utf8', (err, data) => {
                    if (err) {
                        resolve({ success: false, message: `Error reading report file: ${err.message}` });
                    } else {
                        resolve({ success: true, report: data });
                    }
                });
            } else {
                resolve({ success: false, message: `Analysis script exited with code ${code}` });
            }
        });
    });
});

