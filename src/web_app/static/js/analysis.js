// =============================================================================
// ATLAS WEB APP - ANALYSIS PAGE JAVASCRIPT
// =============================================================================
// This script handles the user interactions and API calls for the analysis.html
// page, enabling file uploads, sequence pasting, and real-time feedback.
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize AOS animations
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        offset: 100
    });

    const fileInput = document.getElementById('fastaFile');
    const dropZone = document.getElementById('dropZone');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameSpan = document.getElementById('fileName');
    const sequenceInput = document.getElementById('sequenceInput');
    const charCountSpan = document.getElementById('charCount');
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    
    let uploadedFile = null;

    // --- Enable/Disable Button based on input ---
    function updateButtonState() {
        const hasFile = uploadedFile !== null;
        const hasText = sequenceInput.value.trim().length > 0;
        startAnalysisBtn.disabled = !(hasFile || hasText);
    }
    
    // --- File Input Handlers ---
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadedFile = file;
            fileNameSpan.textContent = file.name;
            fileInfo.classList.remove('hidden');
            sequenceInput.value = ''; // Clear text input if a file is selected
        } else {
            uploadedFile = null;
            fileInfo.classList.add('hidden');
        }
        updateButtonState();
    });

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-primary');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.fasta') || file.name.endsWith('.fa') || file.name.endsWith('.txt')) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else {
            alert('Please drop a valid FASTA file (.fasta, .fa, .txt).');
        }
    });

    // --- Textarea Input Handlers ---
    sequenceInput.addEventListener('input', () => {
        charCountSpan.textContent = `${sequenceInput.value.length} characters`;
        if (sequenceInput.value.trim().length > 0) {
            uploadedFile = null; // Clear file selection if text is entered
            fileInput.value = '';
            fileInfo.classList.add('hidden');
        }
        updateButtonState();
    });

    // --- Analysis Start Button Handler ---
    startAnalysisBtn.addEventListener('click', async () => {
        const hasFile = uploadedFile !== null;
        const hasText = sequenceInput.value.trim().length > 0;

        if (!hasFile && !hasText) {
            alert('Please upload a FASTA file or paste sequences.');
            return;
        }

        let sequences = '';

        if (hasFile) {
            sequences = await uploadedFile.text();
        } else {
            sequences = sequenceInput.value;
        }
        
        // Simple validation to check for a FASTA-like format
        if (!sequences.startsWith('>')) {
            alert('Input does not appear to be in valid FASTA format. Please ensure each sequence starts with a ">" header.');
            return;
        }

        // Show a loading screen or progress indicator
        const uploadSection = document.getElementById('uploadSection');
        const analysisProgressSection = document.getElementById('analysisProgress');
        
        if (uploadSection && analysisProgressSection) {
            // Hide the upload section and show the progress section
            uploadSection.classList.add('hidden');
            analysisProgressSection.classList.remove('hidden');

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequences: sequences })
                });

                const result = await response.json();
                
                if (response.ok) {
                    // Display a success message or trigger a download
                    alert('Analysis complete! Your report is being downloaded.');
                    
                    // Trigger download of the report
                    if (result.report_path) {
                        window.open(`/reports/${result.report_path.split('/').pop()}`, '_blank');
                    }
                    
                    // For demo purposes, reload to restart
                    window.location.reload();

                } else {
                    console.error('Analysis failed:', result.error);
                    alert(`Analysis failed: ${result.error}`);
                    // Hide progress and show upload section again
                    analysisProgressSection.classList.add('hidden');
                    uploadSection.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Network or server error:', error);
                alert(`A network or server error occurred: ${error}`);
                // Hide progress and show upload section again
                analysisProgressSection.classList.add('hidden');
                uploadSection.classList.remove('hidden');
            }
        }
    });
});
