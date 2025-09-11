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
        if (file && (file.name.endsWith('.fasta') || file.name.endsWith('.fa') || file.name.endsWith('.fas') || file.name.endsWith('.txt'))) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else {
            // Using a custom modal would be better, but for now, an alert is a quick fix
            alert('Please drop a valid FASTA file (.fasta, .fa, .fas, .txt).');
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
            alert('Input does not appear to be in valid FASTA format. A FASTA file starts with a ">" followed by a sequence header, and then the DNA sequence on subsequent lines.');
            return;
        }

        const uploadSection = document.getElementById('uploadSection');
        const analysisProgressSection = document.getElementById('analysisProgress');
        
        if (uploadSection && analysisProgressSection) {
            uploadSection.classList.add('hidden');
            analysisProgressSection.classList.remove('hidden');

            const processingStep = document.getElementById('processingStep');
            const aiStep = document.getElementById('aiStep');
            const reportStep = document.getElementById('reportStep');

            // Set initial progress state
            processingStep.classList.add('bg-blue-50');
            processingStep.classList.remove('bg-gray-50', 'opacity-50');
            aiStep.classList.add('bg-gray-50', 'opacity-50');
            reportStep.classList.add('bg-gray-50', 'opacity-50');

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequences: sequences })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

                // Simulate AI Analysis step visually
                processingStep.classList.remove('bg-blue-50');
                processingStep.classList.add('bg-green-50');
                processingStep.querySelector('i').classList.remove('fa-spinner', 'animate-spin');
                processingStep.querySelector('i').classList.add('fa-check');
                processingStep.querySelector('span').textContent = '100%';
                
                aiStep.classList.add('bg-blue-50');
                aiStep.classList.remove('bg-gray-50', 'opacity-50');
                aiStep.querySelector('i').classList.add('fa-spinner', 'animate-spin');
                aiStep.querySelector('span').textContent = 'In Progress';
                
                const result = await response.json();

                if (result.status === 'success') {
                    // Simulate Report Generation step visually
                    aiStep.classList.remove('bg-blue-50');
                    aiStep.classList.add('bg-green-50');
                    aiStep.querySelector('i').classList.remove('fa-spinner', 'animate-spin');
                    aiStep.querySelector('i').classList.add('fa-check');
                    aiStep.querySelector('span').textContent = '100%';

                    reportStep.classList.add('bg-blue-50');
                    reportStep.classList.remove('bg-gray-50', 'opacity-50');
                    reportStep.querySelector('i').classList.add('fa-spinner', 'animate-spin');
                    reportStep.querySelector('span').textContent = 'In Progress';

                    // Finalize the process
                    reportStep.classList.remove('bg-blue-50');
                    reportStep.classList.add('bg-green-50');
                    reportStep.querySelector('i').classList.remove('fa-spinner', 'animate-spin');
                    reportStep.querySelector('i').classList.add('fa-check');
                    reportStep.querySelector('span').textContent = '100%';

                    alert('Analysis complete! Your report is being downloaded.');
                    
                    if (result.report_path) {
                        window.open(`/reports/${result.report_path.split('/').pop()}`, '_blank');
                    }
                    window.location.reload();

                } else {
                    throw new Error(result.error || 'Unknown analysis error.');
                }
            } catch (error) {
                console.error('Analysis failed:', error);
                alert(`Analysis failed: ${error.message}`);
                // Hide progress and show upload section again
                analysisProgressSection.classList.add('hidden');
                uploadSection.classList.remove('hidden');
            }
        }
    });
});
