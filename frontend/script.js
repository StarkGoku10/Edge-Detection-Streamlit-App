/**
 * Edge Detection Studio - Frontend Application
 * Modern Vanilla JavaScript with Component-like Architecture
 */

class EdgeDetectionApp {
    constructor() {
        // Application state
        this.state = {
            currentView: 'upload', // 'upload', 'processing', 'results', 'error'
            uploadProgress: 0,
            jobId: null,
            jobStatus: 'pending',
            originalImage: null,
            processedImageUrl: null,
            error: null,
            isUploading: false,
            isProcessing: false
        };

        // API configuration
        this.api = {
            baseUrl: '', // Same origin, so empty base URL
            endpoints: {
                upload: '/api/upload',
                status: (jobId) => `/api/status/${jobId}`,
                result: (jobId) => `/api/result/${jobId}`
            }
        };

        // UI element references
        this.elements = {};
        
        // Polling configuration
        this.polling = {
            interval: null,
            intervalMs: 2000, // Poll every 2 seconds
            maxAttempts: 300   // Max 10 minutes (300 * 2s)
        };

        // Progress animation interval
        this.progressInterval = null;

        // File selection debounce
        this.fileSelectionInProgress = false;
        
        // Upload zone click debounce
        this.uploadZoneClickInProgress = false;
        this.uploadZoneClickTimeout = null;

        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.cacheElements();
        this.bindEvents();
        this.showView('upload');
        this.hideLoadingOverlay();
    }

    /**
     * Cache DOM element references
     */
    cacheElements() {
        console.log('Caching DOM elements...');
        this.elements = {
            // Sections
            uploadSection: document.getElementById('uploadSection'),
            statusSection: document.getElementById('statusSection'),
            resultsSection: document.getElementById('resultsSection'),
            errorSection: document.getElementById('errorSection'),
            
            // Upload elements
            uploadZone: document.getElementById('uploadZone'),
            fileInput: document.getElementById('fileInput'),
            progressContainer: document.getElementById('progressContainer'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            
            // Status elements
            statusTitle: document.getElementById('statusTitle'),
            statusMessage: document.getElementById('statusMessage'),
            jobId: document.getElementById('jobId'),
            jobStatus: document.getElementById('jobStatus'),
            statusIndicator: document.getElementById('statusIndicator'),
            
            // Results elements
            originalImage: document.getElementById('originalImage'),
            processedImage: document.getElementById('processedImage'),
            originalInfo: document.getElementById('originalInfo'),
            processedInfo: document.getElementById('processedInfo'),
            downloadBtn: document.getElementById('downloadBtn'),
            processAnotherBtn: document.getElementById('processAnotherBtn'),
            
            // Error elements
            errorMessage: document.getElementById('errorMessage'),
            retryBtn: document.getElementById('retryBtn'),
            
            // Loading overlay
            loadingOverlay: document.getElementById('loadingOverlay')
        };
        
        // Verify critical elements exist
        if (!this.elements.fileInput) {
            console.error('File input element not found!');
        }
        if (!this.elements.uploadZone) {
            console.error('Upload zone element not found!');
        }
        
        console.log('DOM elements cached:', Object.keys(this.elements));
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // File upload events - use 'once' option to prevent multiple listeners
        this.elements.uploadZone.addEventListener('click', (e) => {
            console.log('Upload zone clicked', {
                target: e.target.tagName,
                currentTarget: e.currentTarget.tagName,
                targetId: e.target.id,
                currentTargetId: e.currentTarget.id,
                timestamp: Date.now()
            });
            
            // Only handle clicks on the upload zone itself or its direct children
            // Don't handle clicks on the file input
            if (e.target.id === 'fileInput') {
                console.log('Click on file input, ignoring');
                return;
            }
            
            // Prevent event bubbling
            e.preventDefault();
            e.stopPropagation();
            
            // Prevent double-clicking
            if (this.uploadZoneClickInProgress) {
                console.log('Upload zone click already in progress, ignoring');
                return;
            }
            
            // Prevent clicking if already uploading or processing
            if (this.state.isUploading || this.state.isProcessing) {
                console.log('Already uploading/processing, ignoring click');
                return;
            }
            
            // Set debounce flag immediately
            this.uploadZoneClickInProgress = true;
            
            // Clear any existing timeout
            if (this.uploadZoneClickTimeout) {
                clearTimeout(this.uploadZoneClickTimeout);
            }
            
            console.log('Triggering file input click');
            
            // Use a small delay to ensure the click event is processed
            setTimeout(() => {
                this.elements.fileInput.click();
            }, 50);
            
            // Reset the debounce flag after a short delay
            this.uploadZoneClickTimeout = setTimeout(() => {
                this.uploadZoneClickInProgress = false;
                console.log('Upload zone click debounce reset');
            }, 1000); // 1 second debounce
        });

        this.elements.fileInput.addEventListener('change', async (e) => {
            console.log('File input change event triggered');
            console.log('Selected files:', e.target.files);
            console.log('Files length:', e.target.files.length);
            
            // Prevent multiple simultaneous file selections
            if (this.fileSelectionInProgress) {
                console.log('File selection already in progress, ignoring');
                return;
            }
            
            // Prevent default form submission behavior
            e.preventDefault();
            
            // Store the file immediately to prevent loss
            const selectedFile = e.target.files[0];
            
            if (!selectedFile) {
                console.log('No file selected');
                return;
            }
            
            this.fileSelectionInProgress = true;
            
            try {
                await this.handleFileSelect(selectedFile);
            } catch (error) {
                console.error('Error in file selection:', error);
                this.showError(`File selection failed: ${error.message}`);
            } finally {
                this.fileSelectionInProgress = false;
            }
        });

        // Drag and drop events
        this.elements.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadZone.classList.add('dragover');
        });

        this.elements.uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.elements.uploadZone.classList.remove('dragover');
        });

        this.elements.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                this.handleFileSelect(file);
            }
        });

        // Button events
        this.elements.downloadBtn.addEventListener('click', () => {
            this.downloadResult();
        });

        this.elements.processAnotherBtn.addEventListener('click', () => {
            this.resetApplication();
        });

        this.elements.retryBtn.addEventListener('click', () => {
            this.resetApplication();
        });

        // Prevent default drag behavior on the entire document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
    }

    /**
     * Update application state and trigger re-render
     */
    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.render();
    }

    /**
     * Handle file selection
     */
    async handleFileSelect(file) {
        console.log('handleFileSelect called with file:', file);
        
        if (!file) {
            console.log('No file provided to handleFileSelect');
            return;
        }

        try {
            // Validate file
            console.log('Validating file:', file.name, file.type, file.size);
            const validation = this.validateFile(file);
            if (!validation.valid) {
                console.log('File validation failed:', validation.message);
                this.showError(validation.message);
                return;
            }

            console.log('File validation passed');

            // Store original image for display
            this.setState({
                originalImage: file,
                currentView: 'upload'
            });

            console.log('Starting file upload...');
            // Start upload
            await this.uploadFile(file);
            console.log('File upload completed');
            
        } catch (error) {
            console.error('Error in handleFileSelect:', error);
            this.showError(`File processing failed: ${error.message}`);
        }
    }

    /**
     * Validate selected file
     */
    validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];

        if (!allowedTypes.includes(file.type)) {
            return {
                valid: false,
                message: 'Please select a valid image file (JPG, PNG, GIF, BMP, WebP)'
            };
        }

        if (file.size > maxSize) {
            return {
                valid: false,
                message: 'File size must be less than 10MB'
            };
        }

        return { valid: true };
    }

    /**
     * Upload file to the API
     */
    async uploadFile(file) {
        console.log('uploadFile called with:', file.name, file.type, file.size);
        
        try {
            console.log('Setting upload state...');
            this.setState({
                isUploading: true,
                uploadProgress: 0
            });

            console.log('Showing progress...');
            this.showProgress();

            console.log('Creating FormData...');
            const formData = new FormData();
            formData.append('file', file);

            console.log('Making fetch request to:', this.api.endpoints.upload);
            const response = await fetch(this.api.endpoints.upload, {
                method: 'POST',
                body: formData
            });

            console.log('Response received:', response.status, response.statusText);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Upload failed with response:', errorText);
                throw new Error(`Upload failed: ${response.statusText} - ${errorText}`);
            }

            console.log('Parsing JSON response...');
            const result = await response.json();
            console.log('Upload result:', result);

            this.setState({
                isUploading: false,
                jobId: result.job_id,
                uploadProgress: 100
            });

            console.log('Upload successful, starting processing...');
            // Hide progress after a brief delay
            setTimeout(() => {
                this.hideProgress();
                this.startProcessing();
            }, 500);

        } catch (error) {
            console.error('Upload error:', error);
            this.setState({
                isUploading: false,
                uploadProgress: 0
            });
            this.hideProgress();
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    /**
     * Start processing phase
     */
    startProcessing() {
        this.setState({
            currentView: 'processing',
            isProcessing: true
        });

        this.showView('processing');
        this.startStatusPolling();
    }

    /**
     * Start polling for job status
     */
    startStatusPolling() {
        let attempts = 0;
        let isPolling = true;

        this.polling.interval = setInterval(async () => {
            // Check if polling was stopped (race condition prevention)
            if (!isPolling || !this.polling.interval) {
                return;
            }

            attempts++;

            try {
                const status = await this.checkJobStatus();
                
                // Double-check polling is still active after async call
                if (!isPolling || !this.polling.interval) {
                    return;
                }
                
                this.setState({
                    jobStatus: status.status
                });

                // Update status display
                this.updateStatusDisplay(status);

                if (status.status === 'completed') {
                    isPolling = false;
                    this.stopStatusPolling();
                    try {
                        await this.handleProcessingComplete(status);
                    } catch (error) {
                        console.error('Error handling completion:', error);
                        this.showError('Failed to load results. Please try again.');
                    }
                } else if (status.status === 'error') {
                    isPolling = false;
                    this.stopStatusPolling();
                    this.showError(status.error || 'Processing failed');
                } else if (attempts >= this.polling.maxAttempts) {
                    isPolling = false;
                    this.stopStatusPolling();
                    this.showError('Processing timeout. Please try again.');
                }

            } catch (error) {
                console.error('Status polling error:', error);
                
                if (attempts >= this.polling.maxAttempts) {
                    isPolling = false;
                    this.stopStatusPolling();
                    this.showError('Unable to check processing status. Please try again.');
                }
            }
        }, this.polling.intervalMs);
    }

    /**
     * Stop status polling
     */
    stopStatusPolling() {
        if (this.polling.interval) {
            clearInterval(this.polling.interval);
            this.polling.interval = null;
        }
    }

    /**
     * Check job status via API
     */
    async checkJobStatus() {
        const response = await fetch(this.api.endpoints.status(this.state.jobId));
        
        if (!response.ok) {
            throw new Error(`Status check failed: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Update status display during processing
     */
    updateStatusDisplay(status) {
        const statusMessages = {
            pending: 'Preparing image for processing...',
            processing: 'Applying Pb-lite edge detection algorithm...',
            completed: 'Processing completed successfully!',
            error: 'An error occurred during processing.'
        };

        this.elements.statusMessage.textContent = statusMessages[status.status] || 'Processing...';
        this.elements.jobId.textContent = this.state.jobId;
        this.elements.jobStatus.textContent = status.status;

        // Update spinner based on status
        if (status.status === 'completed') {
            this.elements.statusIndicator.innerHTML = '<div class="success-icon">✅</div>';
        } else if (status.status === 'error') {
            this.elements.statusIndicator.innerHTML = '<div class="error-icon">❌</div>';
        }
    }

    /**
     * Handle processing completion
     */
    async handleProcessingComplete(status) {
        try {
            this.setState({
                isProcessing: false,
                processedImageUrl: status.result_url
            });

            // Load and display images
            await this.loadImages();
            
            this.setState({
                currentView: 'results'
            });

            this.showView('results');

        } catch (error) {
            console.error('Error handling completion:', error);
            this.showError('Failed to load results. Please try again.');
        }
    }

    /**
     * Load and display original and processed images
     */
    async loadImages() {
        try {
            // Load original image
            if (this.state.originalImage) {
                const originalUrl = URL.createObjectURL(this.state.originalImage);
                
                await new Promise((resolve, reject) => {
                    this.elements.originalImage.onload = () => {
                        const overlay = this.elements.originalImage.parentElement.querySelector('.image-overlay');
                        if (overlay) {
                            overlay.style.opacity = '0';
                        }
                        resolve();
                    };
                    
                    this.elements.originalImage.onerror = () => {
                        reject(new Error('Failed to load original image'));
                    };
                    
                    this.elements.originalImage.src = originalUrl;
                });

                // Update image info
                this.elements.originalInfo.textContent = this.formatFileSize(this.state.originalImage.size);
            }

            // Load processed image
            if (this.state.processedImageUrl) {
                const processedUrl = this.api.baseUrl + this.state.processedImageUrl;
                
                await new Promise((resolve, reject) => {
                    this.elements.processedImage.onload = () => {
                        const overlay = this.elements.processedImage.parentElement.querySelector('.image-overlay');
                        if (overlay) {
                            overlay.style.opacity = '0';
                        }
                        this.elements.downloadBtn.disabled = false;
                        resolve();
                    };
                    
                    this.elements.processedImage.onerror = () => {
                        reject(new Error('Failed to load processed image'));
                    };
                    
                    this.elements.processedImage.src = processedUrl;
                });

                this.elements.processedInfo.textContent = 'Edge Detection Result';
            }
        } catch (error) {
            console.error('Image loading error:', error);
            throw error; // Re-throw to be handled by caller
        }
    }

    /**
     * Download the processed result
     */
    downloadResult() {
        if (!this.state.processedImageUrl) return;

        const link = document.createElement('a');
        link.href = this.api.baseUrl + this.state.processedImageUrl;
        link.download = 'edge_detection_result.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    /**
     * Show error state
     */
    showError(message) {
        this.setState({
            currentView: 'error',
            error: message,
            isUploading: false,
            isProcessing: false
        });

        this.elements.errorMessage.textContent = message;
        this.stopStatusPolling();
        this.hideProgress();
        this.showView('error');
    }

    /**
     * Reset application to initial state
     */
    resetApplication() {
        console.log('resetApplication called');
        this.stopStatusPolling();
        this.clearProgressInterval();
        
        // Reset debounce flags
        this.fileSelectionInProgress = false;
        this.uploadZoneClickInProgress = false;
        if (this.uploadZoneClickTimeout) {
            clearTimeout(this.uploadZoneClickTimeout);
            this.uploadZoneClickTimeout = null;
        }
        
        this.setState({
            currentView: 'upload',
            uploadProgress: 0,
            jobId: null,
            jobStatus: 'pending',
            originalImage: null,
            processedImageUrl: null,
            error: null,
            isUploading: false,
            isProcessing: false
        });

        // Reset UI elements
        this.elements.fileInput.value = '';
        this.elements.originalImage.src = '';
        this.elements.processedImage.src = '';
        this.elements.downloadBtn.disabled = true;
        this.elements.progressFill.style.width = '0%';

        // Reset image overlays
        document.querySelectorAll('.image-overlay').forEach(overlay => {
            overlay.style.opacity = '1';
        });

        this.showView('upload');
    }

    /**
     * Show specific view section
     */
    showView(viewName) {
        const sections = ['upload', 'status', 'results', 'error'];
        
        sections.forEach(section => {
            const element = this.elements[section + 'Section'];
            if (element) {
                element.style.display = section === viewName ? 'block' : 'none';
            }
        });

        // Special handling for processing view
        if (viewName === 'processing') {
            this.elements.statusSection.style.display = 'block';
        }
    }

    /**
     * Show upload progress
     */
    showProgress() {
        this.elements.progressContainer.style.display = 'block';
        this.updateProgress();
    }

    /**
     * Hide upload progress
     */
    hideProgress() {
        this.elements.progressContainer.style.display = 'none';
        this.clearProgressInterval();
    }

    /**
     * Update progress bar
     */
    updateProgress() {
        this.elements.progressFill.style.width = `${this.state.uploadProgress}%`;
        
        if (this.state.isUploading) {
            // Clear any existing progress interval
            this.clearProgressInterval();
            
            // Simulate progress for visual feedback
            this.progressInterval = setInterval(() => {
                if (this.state.uploadProgress < 90) {
                    this.setState({
                        uploadProgress: this.state.uploadProgress + Math.random() * 10
                    });
                    this.elements.progressFill.style.width = `${this.state.uploadProgress}%`;
                } else {
                    this.clearProgressInterval();
                }
            }, 100);
        }
    }

    /**
     * Clear progress interval to prevent memory leaks
     */
    clearProgressInterval() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    /**
     * Show loading overlay
     */
    showLoadingOverlay() {
        this.elements.loadingOverlay.classList.add('active');
    }

    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        this.elements.loadingOverlay.classList.remove('active');
    }

    /**
     * Render method for state-driven UI updates
     */
    render() {
        // This method can be extended for more complex state-driven rendering
        // For now, most rendering is handled by specific methods
    }

    /**
     * Utility method to format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

/**
 * Application initialization
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the application
    window.edgeDetectionApp = new EdgeDetectionApp();
    
    // Add some visual feedback for better UX
    document.body.addEventListener('click', (e) => {
        if (e.target.matches('.btn')) {
            e.target.style.transform = 'scale(0.98)';
            setTimeout(() => {
                e.target.style.transform = '';
            }, 150);
        }
    });

    // Add keyboard support
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            // Close any modals or reset on escape
            if (window.edgeDetectionApp.state.currentView === 'error') {
                window.edgeDetectionApp.resetApplication();
            }
        }
    });

    console.log('Edge Detection Studio initialized successfully! 🚀');
});

/**
 * Service Worker registration for better caching (optional)
 */
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment if you add a service worker
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered: ', registration))
        //     .catch(registrationError => console.log('SW registration failed: ', registrationError));
    });
} 