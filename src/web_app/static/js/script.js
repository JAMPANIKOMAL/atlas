// Loading Screen Management
window.addEventListener('load', function() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        setTimeout(() => {
            loadingScreen.classList.add('fade-out');
            setTimeout(() => {
                loadingScreen.remove();
            }, 500);
        }, 1000); // Show loading for at least 1 second
    }
});

// Initialize AOS (Animate On Scroll)
document.addEventListener('DOMContentLoaded', function() {
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        offset: 100,
        delay: 0
    });
});

// Smooth scrolling for navigation links
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

// Active navigation highlighting
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

const observerOptions = {
    root: null,
    rootMargin: '-50% 0px -50% 0px',
    threshold: 0
};

const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const currentSection = entry.target.getAttribute('id');
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${currentSection}`) {
                    link.classList.add('active');
                }
            });
        }
    });
}, observerOptions);

sections.forEach(section => {
    navObserver.observe(section);
});

// Legacy scroll animations for compatibility
const fadeElements = document.querySelectorAll('.fade-in');

const fadeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
});

fadeElements.forEach(element => {
    fadeObserver.observe(element);
});

// Counter Animation
function animateCounter(element, target, suffix = '') {
    let current = 0;
    const increment = target / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        
        if (suffix === 'M+') {
            element.textContent = Math.floor(current) + 'M+';
        } else if (suffix === '%') {
            element.textContent = current.toFixed(1) + '%';
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, 20);
}

// Counter Observer
const counters = document.querySelectorAll('.counter');
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const target = parseFloat(entry.target.getAttribute('data-target'));
            const text = entry.target.textContent;
            
            if (text.includes('M+')) {
                animateCounter(entry.target, target, 'M+');
            } else if (text.includes('%')) {
                animateCounter(entry.target, target, '%');
            } else {
                animateCounter(entry.target, target);
            }
            
            counterObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

counters.forEach(counter => {
    counterObserver.observe(counter);
});

// Interactive Timeline Animation System
const processSection = document.getElementById('process');
const timelineSteps = document.querySelectorAll('.timeline-step');
const timelineTracker = document.getElementById('timeline-tracker');
const trackerIcon = document.getElementById('tracker-icon');
const processProgress = document.getElementById('process-progress');

// Step icons mapping
const stepIcons = {
    1: 'fas fa-flask',
    2: 'fas fa-dna', 
    3: 'fas fa-brain',
    4: 'fas fa-chart-line'
};

// Step colors mapping
const stepColors = {
    1: 'border-primary',
    2: 'border-accent', 
    3: 'border-purple-500',
    4: 'border-emerald-500'
};

// Timeline Scroll Tracker
function updateTimelineTracker() {
    if (!processSection || !timelineTracker || !trackerIcon) return;
    
    const sectionRect = processSection.getBoundingClientRect();
    const sectionTop = sectionRect.top + window.pageYOffset;
    const sectionHeight = sectionRect.height;
    const scrollTop = window.pageYOffset;
    const windowHeight = window.innerHeight;
    
    // Calculate when section starts and ends in viewport
    const sectionStart = sectionTop - windowHeight * 0.3;
    const sectionEnd = sectionTop + sectionHeight - windowHeight * 0.7;
    
    // Calculate progress through the section (0 to 1)
    const sectionProgress = Math.max(0, Math.min(1, 
        (scrollTop - sectionStart) / (sectionEnd - sectionStart)
    ));
    
    // Update tracker position (0% to 100% of timeline height to match dotted line)
    const trackerPosition = sectionProgress * 100;
    timelineTracker.style.top = `${trackerPosition}%`;
    
    // More precise step detection using actual step positions
    let activeStep = 1;
    
    // Get actual positions of timeline steps for better synchronization
    const steps = document.querySelectorAll('.timeline-step');
    if (steps.length > 0) {
        const viewportCenter = scrollTop + windowHeight / 2;
        
        steps.forEach((step, index) => {
            const stepRect = step.getBoundingClientRect();
            const stepCenter = stepRect.top + window.pageYOffset + stepRect.height / 2;
            
            if (viewportCenter >= stepCenter - 200) {
                activeStep = index + 1;
            }
        });
    } else {
        // Fallback to progress-based detection
        if (sectionProgress > 0.75) activeStep = 4;
        else if (sectionProgress > 0.5) activeStep = 3;
        else if (sectionProgress > 0.25) activeStep = 2;
    }
    
    // Update tracker icon and border color
    updateTrackerIcon(activeStep);
}

// Function to update tracker icon and styling
let currentStep = 1;
function updateTrackerIcon(stepNumber) {
    if (!trackerIcon || !timelineTracker || currentStep === stepNumber) return;
    
    currentStep = stepNumber;
    
    // Add smooth transition class
    trackerIcon.style.transform = 'scale(0.8)';
    
    setTimeout(() => {
        // Remove all existing icon classes
        trackerIcon.className = '';
        
        // Add new icon class with transitions
        trackerIcon.className = stepIcons[stepNumber] + ' text-white text-sm transition-all duration-300 ease-in-out';
        
        // Remove all border color classes
        timelineTracker.classList.remove('border-primary', 'border-accent', 'border-purple-500', 'border-emerald-500');
        
        // Add appropriate border color
        timelineTracker.classList.add(stepColors[stepNumber]);
        
        // Scale back up
        trackerIcon.style.transform = 'scale(1)';
    }, 150);
}

// Timeline Step Observer for Active State
const timelineObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        const stepNumber = parseInt(entry.target.getAttribute('data-step'));
        
        if (entry.isIntersecting) {
            // Add active class to current step
            entry.target.classList.add('active');
            
            // Remove active class from other steps
            timelineSteps.forEach((step, index) => {
                if (step !== entry.target) {
                    step.classList.remove('active');
                }
            });
            
            // Trigger step-specific animations
            animateStepContent(entry.target, stepNumber);
        }
    });
}, {
    threshold: 0.4,
    rootMargin: '-10% 0px -10% 0px'
});

// Animate step content based on step number
function animateStepContent(step, stepNumber) {
    const content = step.querySelector('.timeline-content');
    const media = step.querySelector('.timeline-media');
    
    // Add subtle content animations
    if (content) {
        content.style.transform = 'scale(1.02)';
        setTimeout(() => {
            if (content) content.style.transform = '';
        }, 300);
    }
}

// Initialize timeline observers
timelineSteps.forEach(step => {
    timelineObserver.observe(step);
});

// Enhanced hover effects for timeline tracker
if (timelineTracker) {
    timelineTracker.addEventListener('mouseenter', () => {
        // Add glow effect to tracker
        timelineTracker.style.boxShadow = '0 0 30px rgba(15, 118, 110, 0.6), 0 0 60px rgba(15, 118, 110, 0.3)';
        timelineTracker.style.transform = 'translateX(-50%) scale(1.1)';
    });
    
    timelineTracker.addEventListener('mouseleave', () => {
        timelineTracker.style.boxShadow = '';
        timelineTracker.style.transform = 'translateX(-50%) scale(1)';
    });
}

// Global Progress Bar Animation
let ticking = false;

function updateProgressBar() {
    const scrollTop = window.pageYOffset;
    const docHeight = document.body.scrollHeight - window.innerHeight;
    const scrollPercent = (scrollTop / docHeight) * 100;
    
    if (processProgress) {
        processProgress.style.width = scrollPercent + '%';
    }
    
    ticking = false;
}

function requestTick() {
    if (!ticking) {
        requestAnimationFrame(() => {
            updateProgressBar();
            updateTimelineTracker();
        });
        ticking = true;
    }
}

window.addEventListener('scroll', requestTick);

// Tooltip System
function createTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    element.appendChild(tooltip);
    return tooltip;
}

// Add tooltips to process cards
processSteps.forEach(step => {
    const card = step.querySelector('.process-card');
    const tooltipText = card.getAttribute('data-tooltip');
    if (tooltipText) {
        createTooltip(card, tooltipText);
    }
});

// Enhanced Hover Effects for Process Cards
processSteps.forEach((step, index) => {
    const card = step.querySelector('.process-card');
    const icon = step.querySelector('.process-icon');
    const badge = step.querySelector('.step-badge');
    
    card.addEventListener('mouseenter', () => {
        // Add pulse animation to badge
        badge.style.animation = 'pulse 1s infinite';
        
        // Rotate icon based on step
        const rotations = ['rotate(15deg)', 'rotate(-15deg)', 'rotate(10deg)', 'rotate(-10deg)'];
        icon.style.transform = rotations[index] || 'rotate(5deg)';
        
        // Add ripple effect to timeline dot
        const dot = step.querySelector('.timeline-dot');
        if (dot) {
            dot.style.animation = 'pulse 1s infinite';
        }
    });
    
    card.addEventListener('mouseleave', () => {
        badge.style.animation = '';
        icon.style.transform = '';
        
        const dot = step.querySelector('.timeline-dot');
        if (dot) {
            dot.style.animation = '';
        }
    });
});

// Feature Cards Enhanced Interactions
const featureCards = document.querySelectorAll('.feature-card');
featureCards.forEach((card, index) => {
    const icon = card.querySelector('.feature-icon i');
    
    card.addEventListener('mouseenter', () => {
        // Staggered icon animations
        const animations = ['bounce', 'pulse', 'rotate', 'shake'];
        const animationType = animations[index % animations.length];
        
        switch(animationType) {
            case 'bounce':
                icon.style.animation = 'iconBounce 0.6s ease-in-out infinite';
                break;
            case 'pulse':
                icon.style.animation = 'pulse 1s ease-in-out infinite';
                break;
            case 'rotate':
                icon.style.animation = 'processIconSpin 1s ease-in-out infinite';
                break;
            case 'shake':
                icon.style.animation = 'iconBounce 0.3s ease-in-out infinite';
                break;
        }
    });
    
    card.addEventListener('mouseleave', () => {
        icon.style.animation = '';
    });
});

// Mobile menu toggle
const mobileMenuBtn = document.getElementById('mobile-menu-btn');

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        // This would typically show/hide a mobile menu
        // For this demo, we'll just add a simple alert
        alert('Mobile menu would open here');
    });
}

// Initialize first section as active
document.addEventListener('DOMContentLoaded', () => {
    const homeLink = document.querySelector('a[href="#home"]');
    if (homeLink) {
        homeLink.classList.add('active');
    }
    
    // Initialize progress bar
    updateProgressBar();
});

// Parallax effect for floating image
const floatingImage = document.querySelector('.floating-image');
if (floatingImage) {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        floatingImage.style.transform = `translateY(${rate}px)`;
    });
}

// Performance optimization: Throttle scroll events
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Apply throttling to scroll-heavy functions
window.addEventListener('scroll', throttle(() => {
    requestTick();
}, 16)); // ~60fps

// Initialize timeline on load
window.addEventListener('load', () => {
    updateTimelineTracker();
});
