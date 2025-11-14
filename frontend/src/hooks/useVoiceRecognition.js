import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook for voice recognition using Web Speech API
 * @returns {Object} Voice recognition state and controls
 */
export const useVoiceRecognition = () => {
    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [error, setError] = useState(null);
    const [isSupported, setIsSupported] = useState(false);
    const recognitionRef = useRef(null);

    useEffect(() => {
        // Check if browser supports Speech Recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            setIsSupported(false);
            setError('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
            return;
        }

        setIsSupported(true);
        
        // Initialize Speech Recognition
        const recognition = new SpeechRecognition();
        recognition.continuous = false; // Stop after user stops speaking
        recognition.interimResults = true; // Show interim results
        recognition.lang = 'en-US'; // Set language

        // Event handlers
        recognition.onstart = () => {
            setIsListening(true);
            setError(null);
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }

            // Update transcript with both interim and final results
            setTranscript(finalTranscript || interimTranscript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            setIsListening(false);
            
            switch (event.error) {
                case 'no-speech':
                    setError('No speech detected. Please try again.');
                    break;
                case 'audio-capture':
                    setError('No microphone found. Please check your microphone settings.');
                    break;
                case 'not-allowed':
                    setError('Microphone permission denied. Please allow microphone access.');
                    break;
                case 'network':
                    setError('Network error. Please check your connection.');
                    break;
                default:
                    setError('Speech recognition error. Please try again.');
            }
        };

        recognition.onend = () => {
            setIsListening(false);
        };

        recognitionRef.current = recognition;

        // Cleanup
        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        };
    }, []);

    const startListening = () => {
        if (!isSupported) {
            setError('Speech recognition is not supported in your browser.');
            return;
        }

        if (recognitionRef.current && !isListening) {
            setTranscript('');
            setError(null);
            try {
                recognitionRef.current.start();
            } catch (err) {
                console.error('Error starting recognition:', err);
                setError('Failed to start voice recognition. Please try again.');
            }
        }
    };

    const stopListening = () => {
        if (recognitionRef.current && isListening) {
            try {
                recognitionRef.current.stop();
            } catch (err) {
                console.error('Error stopping recognition:', err);
            }
        }
    };

    const resetTranscript = () => {
        setTranscript('');
        setError(null);
    };

    return {
        isListening,
        transcript,
        error,
        isSupported,
        startListening,
        stopListening,
        resetTranscript,
    };
};

