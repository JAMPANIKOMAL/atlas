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
        
        // Simulate analysis progress
        simulateAnalysis();
    }

    function simulateAnalysis() {
        let progress = 45;
        const progressElement = document.getElementById('processingProgress');
        
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 100) progress = 100;
            
            progressElement.textContent = `${Math.floor(progress)}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
                completeProcessing();
            }
        }, 500);
    }

    function completeProcessing() {
        // Update processing step
        const processingStep = document.querySelector('.bg-blue-50');
        processingStep.classList.remove('bg-blue-50');
        processingStep.classList.add('bg-green-50');
        
        const processingIcon = processingStep.querySelector('.bg-blue-500');
        processingIcon.classList.remove('bg-blue-500', 'animate-spin');
        processingIcon.classList.add('bg-green-500');
        processingIcon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
        
        const processingProgress = processingStep.querySelector('#processingProgress');
        processingProgress.textContent = '100%';
        processingProgress.classList.remove('text-blue-600');
        processingProgress.classList.add('text-green-600');
        
        // Start AI analysis
        setTimeout(() => {
            startAIAnalysis();
        }, 1000);
    }

    function startAIAnalysis() {
        const aiStep = document.querySelectorAll('.bg-gray-50')[1];
        aiStep.classList.remove('bg-gray-50', 'opacity-50');
        aiStep.classList.add('bg-purple-50');
        
        const aiIcon = aiStep.querySelector('.bg-gray-300');
        aiIcon.classList.remove('bg-gray-300');
        aiIcon.classList.add('bg-purple-500', 'animate-pulse');
        
        const aiProgress = aiStep.querySelector('.text-gray-500');
        aiProgress.textContent = 'Processing...';
        aiProgress.classList.remove('text-gray-500');
        aiProgress.classList.add('text-purple-600');
        
        // Complete AI analysis after delay
        setTimeout(() => {
            completeAIAnalysis();
        }, 3000);
    }

    function completeAIAnalysis() {
        const aiStep = document.querySelectorAll('.bg-purple-50')[0];
        aiStep.classList.remove('bg-purple-50');
        aiStep.classList.add('bg-green-50');
        
        const aiIcon = aiStep.querySelector('.bg-purple-500');
        aiIcon.classList.remove('bg-purple-500', 'animate-pulse');
        aiIcon.classList.add('bg-green-500');
        aiIcon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
        
        const aiProgress = aiStep.querySelector('.text-purple-600');
        aiProgress.textContent = '100%';
        aiProgress.classList.remove('text-purple-600');
        aiProgress.classList.add('text-green-600');
        
        // Start results generation
        setTimeout(() => {
            startResultsGeneration();
        }, 1000);
    }

    function startResultsGeneration() {
        const resultsStep = document.querySelectorAll('.bg-gray-50')[0];
        resultsStep.classList.remove('bg-gray-50', 'opacity-50');
        resultsStep.classList.add('bg-blue-50');
        
        const resultsIcon = resultsStep.querySelector('.bg-gray-300');
        resultsIcon.classList.remove('bg-gray-300');
        resultsIcon.classList.add('bg-blue-500', 'animate-spin');
        resultsIcon.innerHTML = '<i class="fas fa-spinner text-white text-sm"></i>';
        
        const resultsProgress = resultsStep.querySelector('.text-gray-500');
        resultsProgress.textContent = 'Generating...';
        resultsProgress.classList.remove('text-gray-500');
        resultsProgress.classList.add('text-blue-600');
        
        // Complete results generation
        setTimeout(() => {
            completeAnalysis();
        }, 2000);
    }

    function completeAnalysis() {
        const resultsStep = document.querySelector('.bg-blue-50');
        resultsStep.classList.remove('bg-blue-50');
        resultsStep.classList.add('bg-green-50');
        
        const resultsIcon = resultsStep.querySelector('.bg-blue-500');
        resultsIcon.classList.remove('bg-blue-500', 'animate-spin');
        resultsIcon.classList.add('bg-green-500');
        resultsIcon.innerHTML = '<i class="fas fa-check text-white text-sm"></i>';
        
        const resultsProgress = resultsStep.querySelector('.text-blue-600');
        resultsProgress.textContent = 'Complete';
        resultsProgress.classList.remove('text-blue-600');
        resultsProgress.classList.add('text-green-600');
        
        // Show completion message
        setTimeout(() => {
            showResults();
        }, 1000);
    }

    function showResults() {
        // Create results notification
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
        
        // Add download button to progress section
        const progressSection = document.querySelector('#analysisProgress .bg-white');
        const downloadButton = document.createElement('div');
        downloadButton.className = 'mt-8 text-center';
        downloadButton.innerHTML = `
            <button class="bg-gradient-to-r from-primary to-accent text-white px-8 py-3 rounded-lg font-semibold hover:shadow-lg transform hover:-translate-y-1 transition-all duration-300">
                <i class="fas fa-download mr-2"></i>
                Download Results
            </button>
            <button class="ml-4 bg-gray-500 text-white px-8 py-3 rounded-lg font-semibold hover:bg-gray-600 transition-colors duration-300">
                View Report
            </button>
        `;
        
        progressSection.appendChild(downloadButton);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 5000);
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
