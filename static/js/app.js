// AI Timestamp Generator - Main JavaScript

// Global utility functions
window.showAlert = function(message, type = 'info', duration = 5000) {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;

    const alertId = 'alert-' + Date.now();
    const alertHtml = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <div class="d-flex align-items-center">
                <i class="bi bi-${getAlertIcon(type)} me-2"></i>
                <span>${message}</span>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    alertsContainer.insertAdjacentHTML('beforeend', alertHtml);

    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                const bsAlert = new bootstrap.Alert(alertElement);
                bsAlert.close();
            }
        }, duration);
    }
};

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle-fill',
        'danger': 'exclamation-triangle-fill',
        'warning': 'exclamation-triangle-fill',
        'info': 'info-circle-fill'
    };
    return icons[type] || 'info-circle-fill';
}

// Global API helper
window.api = {
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Network error' }));
                throw new Error(error.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    async get(url) {
        return this.request(url);
    },

    async post(url, data) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    async delete(url) {
        return this.request(url, {
            method: 'DELETE'
        });
    }
};

// Global utilities
window.utils = {
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    },

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffDays === 0) {
            return 'Today ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (diffDays === 1) {
            return 'Yesterday ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (diffDays < 7) {
            return `${diffDays} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    copyToClipboard(text) {
        if (navigator.clipboard) {
            return navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            const result = document.execCommand('copy');
            document.body.removeChild(textArea);
            return Promise.resolve(result);
        }
    },

    downloadFile(content, filename, mimeType = 'text/plain') {
        const blob = new Blob([content], { type: mimeType });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
};

// Status checking utility
window.StatusChecker = class {
    constructor(videoId, callback, interval = 5000) {
        this.videoId = videoId;
        this.callback = callback;
        this.interval = interval;
        this.intervalId = null;
        this.isRunning = false;
    }

    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.checkStatus(); // Check immediately
        this.intervalId = setInterval(() => {
            this.checkStatus();
        }, this.interval);

        // Pause when page is hidden
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pause();
            } else {
                this.resume();
            }
        });
    }

    async checkStatus() {
        try {
            const status = await api.get(`/api/video/${this.videoId}/status`);
            this.callback(status);

            // Stop polling if processing is complete
            if (status.status === 'completed' || status.status === 'failed') {
                this.stop();
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }

    pause() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    resume() {
        if (this.isRunning && !this.intervalId) {
            this.intervalId = setInterval(() => {
                this.checkStatus();
            }, this.interval);
        }
    }

    stop() {
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
};

// Loading state manager
window.LoadingManager = class {
    constructor(element, options = {}) {
        this.element = element;
        this.options = {
            text: 'Loading...',
            spinner: true,
            overlay: false,
            ...options
        };
        this.isLoading = false;
        this.originalContent = null;
    }

    show() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.originalContent = this.element.innerHTML;

        const spinnerHtml = this.options.spinner ? 
            '<div class="spinner-border spinner-border-sm me-2"></div>' : '';

        const loadingHtml = `
            <div class="d-flex align-items-center justify-content-center py-3">
                ${spinnerHtml}
                <span>${this.options.text}</span>
            </div>
        `;

        if (this.options.overlay) {
            const overlayHtml = `
                <div class="position-relative">
                    ${this.originalContent}
                    <div class="position-absolute top-0 start-0 w-100 h-100 bg-white bg-opacity-75 d-flex align-items-center justify-content-center">
                        ${loadingHtml}
                    </div>
                </div>
            `;
            this.element.innerHTML = overlayHtml;
        } else {
            this.element.innerHTML = loadingHtml;
        }
    }

    hide() {
        if (!this.isLoading) return;
        
        this.isLoading = false;
        if (this.originalContent) {
            this.element.innerHTML = this.originalContent;
            this.originalContent = null;
        }
    }
};

// Form validation utility
window.FormValidator = class {
    constructor(form) {
        this.form = form;
        this.rules = new Map();
        this.errors = new Map();
    }

    addRule(fieldName, validator, message) {
        if (!this.rules.has(fieldName)) {
            this.rules.set(fieldName, []);
        }
        this.rules.get(fieldName).push({ validator, message });
    }

    validate() {
        this.errors.clear();
        let isValid = true;

        for (const [fieldName, rules] of this.rules) {
            const field = this.form.querySelector(`[name="${fieldName}"]`);
            if (!field) continue;

            for (const rule of rules) {
                if (!rule.validator(field.value, field)) {
                    this.errors.set(fieldName, rule.message);
                    isValid = false;
                    break;
                }
            }
        }

        this.displayErrors();
        return isValid;
    }

    displayErrors() {
        // Clear previous errors
        this.form.querySelectorAll('.is-invalid').forEach(field => {
            field.classList.remove('is-invalid');
        });
        this.form.querySelectorAll('.invalid-feedback').forEach(feedback => {
            feedback.remove();
        });

        // Display new errors
        for (const [fieldName, message] of this.errors) {
            const field = this.form.querySelector(`[name="${fieldName}"]`);
            if (field) {
                field.classList.add('is-invalid');
                const feedback = document.createElement('div');
                feedback.className = 'invalid-feedback';
                feedback.textContent = message;
                field.parentNode.appendChild(feedback);
            }
        }
    }

    clearErrors() {
        this.errors.clear();
        this.displayErrors();
    }
};

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => 
        new bootstrap.Tooltip(tooltipTriggerEl)
    );

    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    const popoverList = [...popoverTriggerList].map(popoverTriggerEl => 
        new bootstrap.Popover(popoverTriggerEl)
    );

    // Handle navigation active states
    const currentPath = window.location.pathname;
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath.includes('/video/') && href.includes('/videos'))) {
            link.classList.add('active');
        }
    });

    // Add fade-in animation to main content
    const main = document.querySelector('main');
    if (main) {
        main.classList.add('fade-in');
    }

    // Handle offline/online events
    window.addEventListener('online', () => {
        showAlert('Connection restored', 'success', 3000);
    });

    window.addEventListener('offline', () => {
        showAlert('No internet connection', 'warning', 0);
    });

    // Handle uncaught errors
    window.addEventListener('error', (event) => {
        console.error('Uncaught error:', event.error);
        showAlert('An unexpected error occurred. Please refresh the page.', 'danger');
    });

    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        showAlert('A network error occurred. Please try again.', 'warning');
    });
});

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showAlert,
        api,
        utils,
        StatusChecker,
        LoadingManager,
        FormValidator
    };
}