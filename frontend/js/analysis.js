// Analysis Page JavaScript Functionality

document.addEventListener('DOMContentLoaded', function() {
    // File upload elements
    const dropZone = document.getElementById('dropZone');
    const fastaFile = document.getElementById('fastaFile');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    
    // Text input elements
    const sequenceInput = document.getElementById('sequenceInput');
    const charCount = document.getElementById('charCount');
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    
    // Progress section
    const analysisProgress = document.getElementById('analysisProgress');
    const processingProgress = document.getElementById('processingProgress');

    // File upload functionality
    fastaFile.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('click', () => fastaFile.click());

    // Text input functionality
    sequenceInput.addEventListener('input', handleTextInput);

    // Analysis button functionality
    startAnalysisBtn.addEventListener('click', startAnalysis);

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            displayFileInfo(file);
            validateInput();
        }
    }

    function handleDragOver(event) {
        event.preventDefault();
        dropZone.classList.add('border-primary', 'bg-primary/5');
    }

    function handleDragLeave(event) {
        event.preventDefault();
        dropZone.classList.remove('border-primary', 'bg-primary/5');
    }

    function handleDrop(event) {
        event.preventDefault();
        dropZone.classList.remove('border-primary', 'bg-primary/5');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            fastaFile.files = files;
            displayFileInfo(file);
            validateInput();
        }
    }

    function displayFileInfo(file) {
        fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        fileInfo.classList.remove('hidden');
        
        // Clear text input when file is uploaded
        sequenceInput.value = '';
        updateCharCount();
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function handleTextInput() {
        updateCharCount();
        validateInput();
        
        // Clear file input when text is entered
        if (sequenceInput.value.trim()) {
            fastaFile.value = '';
            fileInfo.classList.add('hidden');
        }
    }

    function updateCharCount() {
        const count = sequenceInput.value.length;
        charCount.textContent = `${count.toLocaleString()} characters`;
    }

    function validateInput() {
        const hasFile = fastaFile.files.length > 0;
        const hasText = sequenceInput.value.trim().length > 0;
        
        startAnalysisBtn.disabled = !(hasFile || hasText);
        
        if (hasFile || hasText) {
            startAnalysisBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            startAnalysisBtn.classList.add('hover:shadow-lg', 'transform', 'hover:-translate-y-1');
        } else {
            startAnalysisBtn.classList.add('opacity-50', 'cursor-not-allowed');
            startAnalysisBtn.classList.remove('hover:shadow-lg', 'transform', 'hover:-translate-y-1');
        }
    }

    function startAnalysis() {
        if (startAnalysisBtn.disabled) return;
        
        // Get analysis mode
        const analysisMode = document.querySelector('input[name="analysisMode"]:checked').value;
        
        // Show progress section
        analysisProgress.classList.remove('hidden');
        analysisProgress.scrollIntoView({ behavior: 'smooth' });
        
        // Disable start button
        startAnalysisBtn.disabled = true;
        startAnalysisBtn.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i>Processing...';
        
        // Upload data and start real analysis
        uploadAndAnalyze(analysisMode);
    }

    async function uploadAndAnalyze(analysisMode) {
        try {
            const formData = new FormData();
            
            // Add file or text sequences
            if (fastaFile.files.length > 0) {
                formData.append('file', fastaFile.files[0]);
            } else if (sequenceInput.value.trim()) {
                formData.append('sequences', sequenceInput.value.trim());
            }
            
            formData.append('analysis_mode', analysisMode);
            
            // Upload data
            const uploadResponse = await fetch('http://localhost:5000/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!uploadResponse.ok) {
                const error = await uploadResponse.json();
                throw new Error(error.error || 'Upload failed');
            }
            
            const uploadResult = await uploadResponse.json();
            const jobId = uploadResult.job_id;
            
            // Start analysis
            const analyzeResponse = await fetch(`http://localhost:5000/api/analyze/${jobId}`, {
                method: 'POST'
            });
            
            if (!analyzeResponse.ok) {
                const error = await analyzeResponse.json();
                throw new Error(error.error || 'Analysis failed to start');
            }
            
            // Start monitoring progress
            monitorProgress(jobId);
            
        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message);
            resetAnalysisButton();
        }
    }

    async function monitorProgress(jobId) {
        const checkInterval = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:5000/api/status/${jobId}`);
                
                if (!response.ok) {
                    throw new Error('Failed to check status');
                }
                
                const status = await response.json();
                updateProgressDisplay(status);
                
                if (status.status === 'completed') {
                    clearInterval(checkInterval);
                    await loadResults(jobId);
                } else if (status.status === 'error') {
                    clearInterval(checkInterval);
                    showError(status.error || 'Analysis failed');
                    resetAnalysisButton();
                }
            } catch (error) {
                console.error('Status check error:', error);
                clearInterval(checkInterval);
                showError('Failed to monitor analysis progress');
                resetAnalysisButton();
            }
        }, 2000); // Check every 2 seconds
    }

    function updateProgressDisplay(status) {
        const progress = status.progress || 0;
        const currentStep = status.current_step || '';
        
        // Update progress based on current step
        if (currentStep.includes('Processing')) {
            updateProcessingStep(progress);
        } else if (currentStep.includes('AI') || currentStep.includes('Model')) {
            completeProcessingStep();
            updateAIStep(progress);
        } else if (currentStep.includes('Results') || currentStep.includes('Generating')) {
            completeProcessingStep();
            completeAIStep();
            updateResultsStep(progress);
        }
    }

    function updateProcessingStep(progress) {
        const processingStep = document.querySelector('.bg-blue-50') || document.querySelectorAll('.bg-gray-50')[2];
        if (processingStep && processingStep.classList.contains('bg-gray-50')) {
            processingStep.classList.remove('bg-gray-50', 'opacity-50');
            processingStep.classList.add('bg-blue-50');
            
            const icon = processingStep.querySelector('.bg-gray-300');
            if (icon) {
                icon.classList.remove('bg-gray-300');
                icon.classList.add('bg-blue-500', 'animate-spin');
                icon.innerHTML = '<i class="fas fa-spinner text-white text-sm"></i>';
            }
        }
        
        const progressElement = document.getElementById('processingProgress');
        if (progressElement) {
            progressElement.textContent = `${Math.floor(progress)}%`;
            progressElement.classList.remove('text-gray-500');
            progressElement.classList.add('text-blue-600');
        }
    }

    function completeProcessingStep() {
        const processingStep = document.querySelector('.bg-blue-50');
        if (processingStep) {
            processingStep.classList.remove('bg-blue-50');
            processingStep.classList.add('bg-green-50');
            
            const icon = processingStep.querySelector('.bg-blue-500');
            if (icon) {
                icon.classList.remove('bg-blue-500', 'animate-spin');
                icon.classList.add('bg-green-500');
                icon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
            }
            
            const progress = processingStep.querySelector('.text-blue-600');
            if (progress) {
                progress.textContent = '100%';
                progress.classList.remove('text-blue-600');
                progress.classList.add('text-green-600');
            }
        }
    }

    function updateAIStep(progress) {
        const aiStep = document.querySelectorAll('.bg-gray-50')[1];
        if (aiStep) {
            aiStep.classList.remove('bg-gray-50', 'opacity-50');
            aiStep.classList.add('bg-purple-50');
            
            const icon = aiStep.querySelector('.bg-gray-300');
            if (icon) {
                icon.classList.remove('bg-gray-300');
                icon.classList.add('bg-purple-500', 'animate-pulse');
                icon.innerHTML = '<i class="fas fa-brain text-white text-sm"></i>';
            }
            
            const progressText = aiStep.querySelector('.text-gray-500');
            if (progressText) {
                progressText.textContent = `${Math.floor(progress)}%`;
                progressText.classList.remove('text-gray-500');
                progressText.classList.add('text-purple-600');
            }
        }
    }

    function completeAIStep() {
        const aiStep = document.querySelector('.bg-purple-50');
        if (aiStep) {
            aiStep.classList.remove('bg-purple-50');
            aiStep.classList.add('bg-green-50');
            
            const icon = aiStep.querySelector('.bg-purple-500');
            if (icon) {
                icon.classList.remove('bg-purple-500', 'animate-pulse');
                icon.classList.add('bg-green-500');
                icon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
            }
            
            const progress = aiStep.querySelector('.text-purple-600');
            if (progress) {
                progress.textContent = '100%';
                progress.classList.remove('text-purple-600');
                progress.classList.add('text-green-600');
            }
        }
    }

    function updateResultsStep(progress) {
        const resultsStep = document.querySelectorAll('.bg-gray-50')[0];
        if (resultsStep) {
            resultsStep.classList.remove('bg-gray-50', 'opacity-50');
            resultsStep.classList.add('bg-blue-50');
            
            const icon = resultsStep.querySelector('.bg-gray-300');
            if (icon) {
                icon.classList.remove('bg-gray-300');
                icon.classList.add('bg-blue-500', 'animate-spin');
                icon.innerHTML = '<i class="fas fa-spinner text-white text-sm"></i>';
            }
            
            const progressText = resultsStep.querySelector('.text-gray-500');
            if (progressText) {
                progressText.textContent = `${Math.floor(progress)}%`;
                progressText.classList.remove('text-gray-500');
                progressText.classList.add('text-blue-600');
            }
        }
    }

    async function loadResults(jobId) {
        try {
            const response = await fetch(`http://localhost:5000/api/results/${jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to load results');
            }
            
            const results = await response.json();
            displayResults(results, jobId);
            
        } catch (error) {
            console.error('Results loading error:', error);
            showError('Failed to load analysis results');
        }
    }

    function displayResults(results, jobId) {
        // Complete all steps
        completeResultsStep();
        
        // Show completion message
        showSuccessNotification();
        
        // Add download buttons with real functionality
        addDownloadButtons(jobId);
        
        // Display results summary
        displayResultsSummary(results);
    }

    function completeResultsStep() {
        const resultsStep = document.querySelector('.bg-blue-50');
        if (resultsStep) {
            resultsStep.classList.remove('bg-blue-50');
            resultsStep.classList.add('bg-green-50');
            
            const icon = resultsStep.querySelector('.bg-blue-500');
            if (icon) {
                icon.classList.remove('bg-blue-500', 'animate-spin');
                icon.classList.add('bg-green-500');
                icon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
            }
            
            const progress = resultsStep.querySelector('.text-blue-600');
            if (progress) {
                progress.textContent = 'Complete';
                progress.classList.remove('text-blue-600');
                progress.classList.add('text-green-600');
            }
        }
    }

    function showSuccessNotification() {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white p-4 rounded-lg shadow-lg z-50 transform translate-x-full transition-transform duration-300';
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-check-circle mr-3"></i>
                <div>
                    <div class="font-semibold">Analysis Complete!</div>
                    <div class="text-sm opacity-90">Results are ready for download</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Slide in notification
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    function addDownloadButtons(jobId) {
        const progressSection = document.querySelector('#analysisProgress .bg-white');
        const existingButtons = progressSection.querySelector('.download-buttons');
        
        if (!existingButtons) {
            const downloadDiv = document.createElement('div');
            downloadDiv.className = 'mt-8 text-center download-buttons';
            downloadDiv.innerHTML = `
                <button onclick="downloadFile('${jobId}', 'classification_report.csv')" class="bg-gradient-to-r from-primary to-accent text-white px-8 py-3 rounded-lg font-semibold hover:shadow-lg transform hover:-translate-y-1 transition-all duration-300">
                    <i class="fas fa-download mr-2"></i>
                    Download CSV Report
                </button>
                <button onclick="downloadFile('${jobId}', 'results.json')" class="ml-4 bg-gray-500 text-white px-8 py-3 rounded-lg font-semibold hover:bg-gray-600 transition-colors duration-300">
                    <i class="fas fa-file-code mr-2"></i>
                    Download JSON
                </button>
            `;
            progressSection.appendChild(downloadDiv);
        }
    }

    // Make downloadFile function global
    window.downloadFile = function(jobId, filename) {
        const url = `http://localhost:5000/api/download/${jobId}/${filename}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    function displayResultsSummary(results) {
        // Create results summary section if it doesn't exist
        let summarySection = document.getElementById('resultsSummary');
        if (!summarySection) {
            summarySection = document.createElement('section');
            summarySection.id = 'resultsSummary';
            summarySection.className = 'py-16 bg-gray-50';
            summarySection.innerHTML = `
                <div class="container mx-auto px-6">
                    <div class="max-w-6xl mx-auto">
                        <h2 class="text-3xl font-bold text-dark mb-8 text-center font-rejouice">Analysis Results</h2>
                        <div id="resultsSummaryContent"></div>
                    </div>
                </div>
            `;
            
            const analysisSection = document.querySelector('#analysisProgress');
            analysisSection.parentNode.insertBefore(summarySection, analysisSection.nextSibling);
        }
        
        // Populate results
        const content = document.getElementById('resultsSummaryContent');
        content.innerHTML = `
            <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                <div class="bg-white rounded-lg p-6 shadow-lg text-center">
                    <div class="text-3xl font-bold text-primary mb-2">${results.input_sequences}</div>
                    <div class="text-gray-600">Total Sequences</div>
                </div>
                <div class="bg-white rounded-lg p-6 shadow-lg text-center">
                    <div class="text-3xl font-bold text-accent mb-2">${results.species_identified}</div>
                    <div class="text-gray-600">Species Identified</div>
                </div>
                <div class="bg-white rounded-lg p-6 shadow-lg text-center">
                    <div class="text-3xl font-bold text-green-600 mb-2">${results.classified_sequences}</div>
                    <div class="text-gray-600">Classified</div>
                </div>
                <div class="bg-white rounded-lg p-6 shadow-lg text-center">
                    <div class="text-3xl font-bold text-purple-600 mb-2">${results.novel_sequences}</div>
                    <div class="text-gray-600">Novel/Unclassified</div>
                </div>
            </div>
            
            <div class="grid md:grid-cols-2 gap-8">
                <div class="bg-white rounded-lg p-6 shadow-lg">
                    <h3 class="text-xl font-bold text-dark mb-4">Top Species Found</h3>
                    <div class="space-y-3">
                        ${results.top_species.map(species => `
                            <div class="flex justify-between items-center p-3 bg-gray-50 rounded">
                                <div>
                                    <div class="font-medium">${species.name}</div>
                                    <div class="text-sm text-gray-600">Confidence: ${(species.confidence * 100).toFixed(1)}%</div>
                                </div>
                                <div class="text-primary font-bold">${species.count}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="bg-white rounded-lg p-6 shadow-lg">
                    <h3 class="text-xl font-bold text-dark mb-4">Diversity Metrics</h3>
                    <div class="space-y-4">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Shannon Diversity:</span>
                            <span class="font-semibold">${results.diversity_metrics.shannon_diversity.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Simpson Diversity:</span>
                            <span class="font-semibold">${results.diversity_metrics.simpson_diversity.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Species Richness:</span>
                            <span class="font-semibold">${results.diversity_metrics.species_richness}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Evenness:</span>
                            <span class="font-semibold">${results.diversity_metrics.evenness.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Scroll to results
        summarySection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-red-500 text-white p-4 rounded-lg shadow-lg z-50 transform translate-x-full transition-transform duration-300';
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-exclamation-triangle mr-3"></i>
                <div>
                    <div class="font-semibold">Analysis Error</div>
                    <div class="text-sm opacity-90">${message}</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Slide in notification
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Remove notification after 8 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 8000);
    }

    function resetAnalysisButton() {
        startAnalysisBtn.disabled = false;
        startAnalysisBtn.innerHTML = '<i class="fas fa-play mr-2"></i>Start Analysis';
        validateInput();
    }

    // Enhanced model card interactions
    const modelCards = document.querySelectorAll('.bg-gradient-to-br');
    modelCards.forEach(card => {
        // Add interactive-card class for enhanced animations
        card.classList.add('interactive-card');
        
        card.addEventListener('mouseenter', () => {
            const icon = card.querySelector('i');
            if (icon) {
                icon.style.transform = 'scale(1.1) rotate(5deg)';
                icon.style.transition = 'all 0.3s ease';
            }
        });
        
        card.addEventListener('mouseleave', () => {
            const icon = card.querySelector('i');
            if (icon) {
                icon.style.transform = 'scale(1) rotate(0deg)';
            }
        });
        
        // Add click animation
        card.addEventListener('click', () => {
            card.style.transform = 'scale(0.98)';
            setTimeout(() => {
                card.style.transform = '';
            }, 150);
        });
    });

    // Enhanced form interactions
    const formElements = document.querySelectorAll('input, textarea, button');
    formElements.forEach(element => {
        // Add focus animations
        element.addEventListener('focus', () => {
            element.classList.add('ring-2', 'ring-primary/20');
        });
        
        element.addEventListener('blur', () => {
            element.classList.remove('ring-2', 'ring-primary/20');
        });
    });

    // Add loading states to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled && !this.classList.contains('loading')) {
                this.classList.add('loading');
                const originalContent = this.innerHTML;
                
                // Add loading spinner for non-analysis buttons
                if (this.id !== 'startAnalysisBtn') {
                    this.innerHTML = '<div class="loading-spinner inline-block mr-2"></div>Processing...';
                    
                    setTimeout(() => {
                        this.innerHTML = originalContent;
                        this.classList.remove('loading');
                    }, 1000);
                }
            }
        });
    });

    // Initialize character count
    updateCharCount();
    validateInput();
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
