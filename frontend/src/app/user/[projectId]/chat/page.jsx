'use client';

import { useState, useRef, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useParams, useSearchParams } from 'next/navigation';
import { sendMessage, addHumanMessage, addAiMessage, clearChat, getChatHistory, markMessageImportant, unmarkMessageImportant, deleteMessage, getImportantMessages, getConversations, generateVisualization } from '@/lib/store/users-panel/chat/chatSlice';
import { FaRobot, FaPaperPlane, FaUser, FaStar, FaChevronDown, FaTrash, FaChartBar, FaMicrophone, FaMicrophoneSlash } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getProjects } from '@/lib/store/users-panel/projects/projectSlice';
import { format, isValid, isToday, isYesterday } from 'date-fns';
import { useVoiceRecognition } from '@/hooks/useVoiceRecognition';
import { toast } from 'react-hot-toast';

// Enhanced Visualization Component with Labels
const VisualizationComponent = ({ visualization }) => {
    console.log('[DEBUG] VisualizationComponent called with:', visualization);
    
    // Handle both direct visualization data and nested structure
    const visualizationData = visualization?.visualization || visualization;
    console.log('[DEBUG] VisualizationComponent - extracted visualizationData:', visualizationData);
    
    if (!visualizationData || !visualizationData.data) {
        console.log('[DEBUG] VisualizationComponent: No visualization data, returning null');
        return null;
    }

    // Get chart type icon and label
    const getChartTypeInfo = (chartType) => {
        const types = {
            'bar': { icon: '📊', label: 'Bar Chart', color: 'blue' },
            'pie': { icon: '🥧', label: 'Pie Chart', color: 'green' },
            'line': { icon: '📈', label: 'Line Chart', color: 'purple' },
            'scatter': { icon: '🔍', label: 'Scatter Plot', color: 'orange' },
            'histogram': { icon: '📊', label: 'Histogram', color: 'red' },
            'chart': { icon: '📊', label: 'Chart', color: 'blue' }
        };
        return types[chartType] || { icon: '📊', label: 'Visualization', color: 'blue' };
    };

    const chartTypeInfo = getChartTypeInfo(visualizationData.type);
    const chartTypeColor = chartTypeInfo.color;

    return (
        <div className="mt-4 p-4 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            {/* Header with Chart Type Label */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                    <span className="text-lg">{chartTypeInfo.icon}</span>
                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        {chartTypeInfo.label}
                    </h4>
                    <span className={`px-2 py-1 text-xs rounded-full bg-${chartTypeColor}-100 text-${chartTypeColor}-800 dark:bg-${chartTypeColor}-900 dark:text-${chartTypeColor}-200`}>
                        {visualizationData.type?.toUpperCase() || 'CHART'}
                    </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                    Generated Chart
                </div>
            </div>

            {/* Visualization Image */}
            <div className="flex justify-center mb-3">
                <div className="relative">
                    <img 
                        src={`data:image/png;base64,${visualizationData.data}`}
                        alt={visualizationData.title || 'Data Visualization'}
                        className="max-w-full h-auto rounded-lg shadow-md border border-gray-200 dark:border-gray-600"
                        style={{ maxHeight: '400px' }}
                    />
                    {/* Overlay label */}
                    <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                        {chartTypeInfo.label}
                    </div>
                </div>
            </div>

            {/* Metadata Section */}
            <div className="space-y-2">
                {visualizationData.title && (
                    <div className="flex items-start space-x-2">
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mt-0.5">Title:</span>
                        <p className="text-xs text-gray-700 dark:text-gray-300 flex-1">
                            {visualizationData.title}
                        </p>
                    </div>
                )}
                
                {visualizationData.query && (
                    <div className="flex items-start space-x-2">
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400 mt-0.5">Query:</span>
                        <p className="text-xs text-gray-600 dark:text-gray-400 flex-1 italic">
                            "{visualizationData.query}"
                        </p>
                    </div>
                )}

                {/* Chart Type Badge */}
                <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-600">
                    <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400">Chart Type:</span>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-${chartTypeColor}-100 text-${chartTypeColor}-800 dark:bg-${chartTypeColor}-900 dark:text-${chartTypeColor}-200`}>
                            {chartTypeInfo.icon} {chartTypeInfo.label}
                        </span>
                    </div>
                    <div className="text-xs text-gray-400 dark:text-gray-500">
                        Interactive
                    </div>
                </div>
            </div>
        </div>
    );
};

// Enhanced CSS for modern LLM-like interface
const modernLLMStyles = `
  .typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #6b7280;
    animation: typing 1.4s infinite ease-in-out;
  }
  
  @keyframes typing {
    0%, 60%, 100% {
      transform: translateY(0);
      opacity: 0.4;
    }
    30% {
      transform: translateY(-8px);
      opacity: 1;
    }
  }

  /* Modern scrollbar - always visible */
  .custom-scrollbar::-webkit-scrollbar {
    width: 8px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background: rgba(156, 163, 175, 0.4);
    border-radius: 4px;
    transition: background 0.2s ease;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: rgba(156, 163, 175, 0.6);
  }
  
  .dark .custom-scrollbar::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
  }
  
  .dark .custom-scrollbar::-webkit-scrollbar-thumb {
    background: rgba(75, 85, 99, 0.4);
  }
  
  .dark .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: rgba(75, 85, 99, 0.6);
  }

  /* Message animations */
  .message-enter {
    opacity: 0;
    transform: translateY(10px);
    animation: messageSlideIn 0.3s ease-out forwards;
  }

  @keyframes messageSlideIn {
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* Input focus effects */
  .chat-input:focus {
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .dark .chat-input:focus {
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }

  /* Hide scrollbar for textarea */
  .chat-input::-webkit-scrollbar {
    display: none;
  }

  .chat-input {
    -ms-overflow-style: none;  /* IE and Edge */
    scrollbar-width: none;  /* Firefox */
  }

  /* Smooth transitions */
  .message-bubble {
    transition: all 0.2s ease;
  }

  .message-bubble:hover {
    transform: translateY(-1px);
  }

  /* Modern button styles */
  .send-button {
    transition: all 0.2s ease;
  }

  .send-button:hover:not(:disabled) {
    transform: scale(1.05);
  }

  .send-button:active {
    transform: scale(0.95);
  }
`;

// Helper function to format text with bold formatting
const formatWithBold = (text) => {
    if (!text || typeof text !== 'string') return text;
    
    let formatted = text;
    
    // Helper to check if text is already wrapped in strong tag
    // Uses a closure to access the current formatted string
    const createBoldChecker = (currentString) => {
        return (offset) => {
            if (!currentString || typeof currentString !== 'string') return false;
            const before = currentString.substring(Math.max(0, offset - 30), offset);
            return before.includes('<strong>') && !before.includes('</strong>');
        };
    };
    
    // Make years bold first (4-digit years like 2022, 2023, etc.)
    const checkBold1 = createBoldChecker(formatted);
    formatted = formatted.replace(/\b(19|20)\d{2}\b/g, (match, p1, offset) => {
        if (!checkBold1(offset)) {
            return `<strong>${match}</strong>`;
        }
        return match;
    });
    
    // Make months bold (January, February, etc.)
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December'];
    months.forEach(month => {
        const regex = new RegExp(`\\b${month}\\b`, 'gi');
        const checkBold2 = createBoldChecker(formatted);
        formatted = formatted.replace(regex, (match, offset) => {
            if (!checkBold2(offset)) {
                return `<strong>${match}</strong>`;
            }
            return match;
        });
    });
    
    // Make other numbers bold (but not years which are already done)
    const checkBold3 = createBoldChecker(formatted);
    formatted = formatted.replace(/\b(\d+)\b/g, (match, p1, offset) => {
        // Skip if it's a 4-digit year (already handled)
        if (/^\d{4}$/.test(match) && (match.startsWith('19') || match.startsWith('20'))) {
            return match;
        }
        if (!checkBold3(offset)) {
            return `<strong>${match}</strong>`;
        }
        return match;
    });
    
    // Make common important words/phrases bold
    const importantPhrases = [
        'employees', 'employee', 'hired', 'hiring', 'months', 'month',
        'analyze', 'analysis', 'factors', 'patterns', 'concentrated',
        'data', 'results', 'total', 'average', 'percentage', 'percent',
        'year', 'years', 'quarter', 'quarters', 'week', 'weeks', 'day', 'days'
    ];
    
    importantPhrases.forEach(phrase => {
        const regex = new RegExp(`\\b${phrase}\\b`, 'gi');
        const checkBold4 = createBoldChecker(formatted);
        formatted = formatted.replace(regex, (match, offset) => {
            if (!checkBold4(offset)) {
                return `<strong>${match}</strong>`;
            }
            return match;
        });
    });
    
    return formatted;
};

// Helper function to convert text into bullet points and extract greeting
const convertToBulletPoints = (text) => {
    if (!text || typeof text !== 'string') return { greeting: null, bulletPoints: [] };
    
    // Split by newlines first to handle multi-line greetings
    let lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    
    // Extract greeting lines from the beginning
    const greetingPatterns = [
        /^👋\s*Hello\s*[!.]?$/i,
        /^Welcome\s+to\s*\([^)]+\)\s*[!.]?$/i,
        /^Welcome\s*to\s*\([^)]+\)\s*[!.]?$/i,
        /^Welcome\s*[!.]?$/i,
        /^Welcome\s+to\s*[!.]?$/i
    ];
    
    // Extract greeting lines
    let greeting = null;
    const greetingLines = [];
    while (lines.length > 0 && greetingPatterns.some(pattern => pattern.test(lines[0]))) {
        greetingLines.push(lines.shift());
    }
    
    // Join greeting lines if any were found
    if (greetingLines.length > 0) {
        greeting = greetingLines.join(' ');
    }
    
    // Also check for greeting at the start of the text (in case it's not on a separate line)
    if (!greeting) {
        const greetingMatch = text.match(/^(👋\s*Hello|Welcome\s+to\s*\([^)]+\)|Welcome\s*to\s*\([^)]+\)|Welcome)\s*[!.]?\s*/i);
        if (greetingMatch) {
            greeting = greetingMatch[0].trim();
        }
    }
    
    // Join back and clean up remaining text
    let cleanedText = lines.join('\n')
        // Also remove any remaining greeting patterns in the text
        .replace(/^👋\s*Hello\s*[!.]?\s*/i, '')
        .replace(/^Welcome\s+to\s*\([^)]+\)\s*[!.]?\s*/i, '')
        .replace(/^Welcome\s*to\s*\([^)]+\)\s*[!.]?\s*/i, '')
        .replace(/^Welcome\s*[!.]?\s*/i, '')
        .replace(/^Welcome\s+to\s*[!.]?\s*/i, '')
        .replace(/^[!.\s]+/, '')
        .trim();
    
    // Re-split after cleaning
    lines = cleanedText.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    
    // If we have multiple lines, use them as bullet points
    if (lines.length > 1) {
        return {
            greeting: greeting,
            bulletPoints: lines.map(line => line.trim()).filter(line => line.length > 0)
        };
    }
    
    // If no newlines, split by sentences (periods, exclamation, question marks)
    // Use a regex that captures the sentence ending punctuation
    const sentences = cleanedText.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
    
    if (sentences.length > 1) {
        return {
            greeting: greeting,
            bulletPoints: sentences.map(s => s.trim()).filter(s => s.length > 0)
        };
    }
    
    // If still only one item, try splitting by common separators (but only if it makes sense)
    if (cleanedText.includes(',') || cleanedText.includes(';')) {
        const parts = cleanedText.split(/[,;]\s+/).filter(p => p.trim().length > 0);
        if (parts.length > 1) {
            return {
                greeting: greeting,
                bulletPoints: parts.map(p => p.trim()).filter(p => p.length > 0)
            };
        }
    }
    
    // If all else fails, return the cleaned text as a single bullet point
    return {
        greeting: greeting,
        bulletPoints: cleanedText ? [cleanedText] : []
    };
};

// Modern LLM-style MessageBubble Component
const MessageBubble = ({ message, aiAgentName, onToggleImportance, isImportant }) => {
    const isAi = message.role === 'ai';
    
    // Modern LLM-style layout
    const containerClasses = isAi ? 'justify-start' : 'justify-end';
    const bubbleAlignmentClasses = isAi ? '' : 'flex-row-reverse';
    
    // Enhanced avatar styling
    const iconContainerClasses = isAi 
        ? 'bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg' 
        : 'bg-gradient-to-br from-gray-500 to-gray-600 shadow-lg';
    
    // Modern bubble styling
    const bubbleClasses = isAi
        ? `message-bubble rounded-2xl bg-white text-gray-900 shadow-sm border border-gray-200 dark:bg-gray-900 dark:text-gray-100 dark:border-gray-700 ${message.isError ? 'border-red-300 dark:border-red-600' : ''} ${isImportant ? 'ring-2 ring-yellow-400 dark:ring-yellow-500' : ''}`
        : `message-bubble rounded-2xl bg-blue-600 text-white shadow-sm ${isImportant ? 'ring-2 ring-yellow-400 dark:ring-yellow-500' : ''}`;
        
    const authorTextClasses = isAi ? 'text-gray-700 dark:text-gray-300' : 'text-blue-100';

    const proseClasses = isAi 
        ? 'prose prose-sm max-w-none prose-p:my-1.5 prose-headings:my-2 prose-ul:my-1.5 prose-ol:my-1.5 dark:prose-invert prose-p:text-gray-800 dark:prose-p:text-gray-200' 
        : 'prose prose-invert prose-sm max-w-none prose-p:my-1.5 prose-headings:my-2 prose-ul:my-1.5 prose-ol:my-1.5';

    const handleStarClick = (e) => {
        e.stopPropagation();
        if (onToggleImportance && message.id) {
            onToggleImportance(message.id, !isImportant);
        }
    };

    // Format date and time according to device's local timezone and locale
    const dateString = message.createdAt || message.created_at;
    
    // Parse date string - handle both ISO format with/without timezone
    let dateObj;
    if (typeof dateString === 'string') {
        // If the string doesn't end with Z or have timezone, assume UTC
        const normalizedDate = dateString.endsWith('Z') || dateString.includes('+') || dateString.includes('-', 10)
            ? dateString
            : dateString + 'Z';
        dateObj = new Date(normalizedDate);
    } else if (dateString instanceof Date) {
        dateObj = dateString;
    } else {
        dateObj = new Date(dateString);
    }
    
    const formattedDateTime = (() => {
        if (!isValid(dateObj) || isNaN(dateObj.getTime())) return '';
        
        const now = new Date();
        const diffInMinutes = Math.floor((now - dateObj) / (1000 * 60));
        
        // Show relative time for recent messages (within last hour)
        if (diffInMinutes < 1) {
            return 'Just now';
        } else if (diffInMinutes < 60) {
            return `${diffInMinutes} min ago`;
        }
        
        // Show time for today's messages
        if (isToday(dateObj)) {
            return format(dateObj, 'h:mm a'); // e.g., "2:30 PM"
        }
        
        // Show "Yesterday" for yesterday's messages
        if (isYesterday(dateObj)) {
            return `Yesterday ${format(dateObj, 'h:mm a')}`;
        }
        
        // Show date and time for older messages
        const diffInDays = Math.floor((now - dateObj) / (1000 * 60 * 60 * 24));
        if (diffInDays < 7) {
            return format(dateObj, 'EEE h:mm a'); // e.g., "Mon 2:30 PM"
        }
        
        // Show full date for older messages
        return format(dateObj, 'MMM d, h:mm a'); // e.g., "Jan 15, 2:30 PM"
    })();

    // Convert AI message content to bullet points and extract greeting
    const { greeting, bulletPoints } = isAi ? convertToBulletPoints(message.content) : { greeting: null, bulletPoints: null };

    return (
        <div className={`message mb-4 flex ${containerClasses} message-enter`}>
            <div className={`flex items-start gap-3 max-w-4xl w-full ${bubbleAlignmentClasses}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${iconContainerClasses}`}>
                    {isAi ? <FaRobot className="text-white text-sm" /> : <FaUser className="text-white text-sm" />}
                </div>
                <div className={`relative px-4 py-3 group ${bubbleClasses}`}>
                    <div className={`text-xs font-medium mb-2 opacity-70 ${authorTextClasses}`}>
                        {isAi ? aiAgentName : 'You'}
                    </div>
                    <div className={proseClasses} style={isAi ? { whiteSpace: 'pre-line' } : {}}>
                        {isAi ? (
                            <>
                                {/* Show greeting as regular text if present */}
                                {greeting && (
                                    <div 
                                        className="mb-3 text-gray-800 dark:text-gray-200 font-medium"
                                        dangerouslySetInnerHTML={{ __html: formatWithBold(greeting) }}
                                    />
                                )}
                                {/* Render AI responses as bullet points with bold formatting */}
                                {bulletPoints && bulletPoints.length > 0 ? (
                                    <ul className="list-disc list-inside space-y-2 text-gray-800 dark:text-gray-200">
                                        {bulletPoints.map((point, index) => (
                                            <li 
                                                key={index} 
                                                className="ml-2"
                                                dangerouslySetInnerHTML={{ __html: formatWithBold(point) }}
                                            />
                                        ))}
                                    </ul>
                                ) : !greeting ? (
                                    <ul className="list-disc list-inside space-y-2 text-gray-800 dark:text-gray-200">
                                        <li dangerouslySetInnerHTML={{ __html: formatWithBold(message.content) }} />
                                    </ul>
                                ) : null}
                            </>
                        ) : (
                            // Render human messages normally
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                            </ReactMarkdown>
                        )}
                        {/* Visualization Component */}
                        {isAi && message.visualization && (
                            <VisualizationComponent visualization={message.visualization} />
                        )}
                    </div>
                    {/* Timestamp and actions */}
                    <div className="flex items-center justify-between mt-3 pt-2 border-t border-gray-100 dark:border-gray-700">
                        <div 
                            className="text-xs text-gray-400 cursor-help" 
                            title={isValid(dateObj) ? format(dateObj, 'PPpp') : ''} // Full date/time on hover
                        >
                            {formattedDateTime}
                        </div>
                        {/* Star Button */}
                        {message.id && message.id !== null && (
                            <button
                                onClick={handleStarClick}
                                className={`p-1.5 rounded-full transition-all duration-200 hover:scale-110 ${
                                    isImportant 
                                        ? 'text-yellow-500 hover:text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20' 
                                        : 'text-gray-400 hover:text-yellow-500 hover:bg-gray-50 dark:hover:bg-gray-800'
                                }`}
                                title={isImportant ? 'Remove from important' : 'Mark as important'}
                            >
                                <FaStar className={`w-3 h-3 ${isImportant ? 'fill-current' : ''}`} />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Helper to display messages in chronological order
function getMessagesInOrder(messages) {
    // Ensure messages are in chronological order by created_at
    const sortedMessages = [...messages].sort((a, b) => {
        const parseDate = (d) => {
            const date = new Date(d);
            return isNaN(date.getTime()) ? new Date('1970-01-01T00:00:00Z') : date;
        };
        const dateA = parseDate(a.createdAt || a.created_at);
        const dateB = parseDate(b.createdAt || b.created_at);
        if (dateA - dateB !== 0) return dateA - dateB;
        // If timestamps are equal, show human before ai
        if (a.role === 'human' && b.role === 'ai') return -1;
        if (a.role === 'ai' && b.role === 'human') return 1;
        // As a last resort, sort by id (string compare)
        return (a.id || '').localeCompare(b.id || '');
    });
    return sortedMessages;
}

// Main ChatPage Component
const ChatPage = () => {
    const [inputValue, setInputValue] = useState('');
    const [showScrollButton, setShowScrollButton] = useState(false);
    const chatContainerRef = useRef(null);
    const inputRef = useRef(null);
    const prevConversationIdRef = useRef(null);
    
    // Voice recognition hook
    const {
        isListening,
        transcript,
        error: voiceError,
        isSupported: isVoiceSupported,
        startListening,
        stopListening,
        resetTranscript,
    } = useVoiceRecognition();

    const dispatch = useDispatch();
    const routerParams = useParams();
    const searchParams = useSearchParams();
    const { projectId } = routerParams;
    
    // Get conversationId from URL query parameters
    const urlConversationId = searchParams.get('conversationId');
    const isNewChat = searchParams.get('new') === '1';

    const { projects, status: projectStatus } = useSelector((state) => state.projects);
    const { messages, status, conversationId, currentConversation, importantMessages = [], error } = useSelector((state) => state.chat);
    const isLoading = status === 'loading'; // Only for AI responses, not importance operations
    
    const currentProject = projects.find(p => p.id === projectId);
    const agentName = currentProject?.bot_name || 'AI Business Agent';
    
    // Use conversation title if available, otherwise fallback to agent name
    const chatHeaderTitle = currentConversation?.title || agentName;
    
    useEffect(() => {
        if (projectStatus === 'idle') {
            dispatch(getProjects());
        }

        // Clear chat if starting a new chat
        if (isNewChat) {
            dispatch(clearChat());
            // Clear input field
            setInputValue('');
            // Reset textarea height
            if (inputRef.current) {
                inputRef.current.style.height = 'auto';
            }
            // Remove the 'new=1' and 't' (timestamp) parameters from URL after clearing
            setTimeout(() => {
                if (typeof window !== 'undefined') {
                    const url = new URL(window.location.href);
                    url.searchParams.delete('new');
                    url.searchParams.delete('t'); // Remove timestamp parameter
                    const newUrl = url.pathname + (url.search ? url.search : '');
                    window.history.replaceState({}, '', newUrl);
                }
            }, 100);
        }

        // Load chat history if conversationId is provided in URL (and not a new chat)
        if (urlConversationId && !isNewChat) {
            dispatch(getChatHistory(urlConversationId));
        }
    }, [projectId, projectStatus, urlConversationId, isNewChat, dispatch]);

    // Always fetch important messages when projectId changes (not conversationId)
    useEffect(() => {
        if (projectId) {
            dispatch(getImportantMessages(projectId));
        }
    }, [projectId, dispatch]);

    // Scroll to bottom when component mounts, conversation changes, or messages load
    useEffect(() => {
        if (chatContainerRef.current && messages.length > 0) {
            // Small delay to ensure DOM is updated
            setTimeout(() => {
                chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
            }, 100);
        }
    }, [urlConversationId, isNewChat, messages]);

    // Handle scroll events to show/hide scroll button
    useEffect(() => {
        const chatContainer = chatContainerRef.current;
        if (!chatContainer) return;

        const handleScroll = () => {
            const { scrollTop, scrollHeight, clientHeight } = chatContainer;
            const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
            setShowScrollButton(!isNearBottom);
        };

        chatContainer.addEventListener('scroll', handleScroll);
        return () => chatContainer.removeEventListener('scroll', handleScroll);
    }, []);

    const scrollToBottom = () => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTo({
                top: chatContainerRef.current.scrollHeight,
                behavior: 'smooth'
            });
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const userMessage = inputValue.trim();
        console.log('[DEBUG] Send message clicked:', {
            userMessage,
            conversationId: conversationId || urlConversationId,
            projectId: projectId
        });
        
        if (!userMessage) {
            console.log('[DEBUG] No user message to send');
            return;
        }

        // Stop voice recognition if it's active
        if (isListening) {
            stopListening();
        }

        dispatch(addHumanMessage({ content: userMessage }));
        dispatch(sendMessage({
            naturalLanguageQuery: userMessage,
            conversationId: conversationId || urlConversationId,
            projectId: projectId
        }));

        setInputValue('');
        resetTranscript();
        // Reset textarea height
        if (inputRef.current) {
            inputRef.current.style.height = 'auto';
        }
    };

    const handleGenerateVisualization = async () => {
        const userMessage = inputValue.trim();
        console.log('[DEBUG] Generate visualization clicked:', {
            userMessage,
            conversationId: conversationId || urlConversationId,
            projectId: projectId
        });
        
        if (!userMessage) {
            console.log('[DEBUG] No user message to visualize');
            return;
        }

        // Stop voice recognition if it's active
        if (isListening) {
            stopListening();
        }

        dispatch(addHumanMessage({ content: userMessage }));
        
        // Add a placeholder AI message for the visualization
        dispatch(addAiMessage({ 
            content: "Generating visualization...", 
            id: `temp-${Date.now()}`,
            visualization: null 
        }));
        
        dispatch(generateVisualization({
            naturalLanguageQuery: userMessage,
            conversationId: conversationId || urlConversationId,
            projectId: projectId
        }));

        setInputValue('');
        resetTranscript();
        // Reset textarea height
        if (inputRef.current) {
            inputRef.current.style.height = 'auto';
        }
    };

    // Auto-resize textarea
    useEffect(() => {
        if (inputRef.current) {
            // Reset height to auto to get the correct scrollHeight
            inputRef.current.style.height = 'auto';
            
            // Calculate new height with a maximum of 128px
            const newHeight = Math.min(inputRef.current.scrollHeight, 128);
            inputRef.current.style.height = newHeight + 'px';
            
            // If content exceeds max height, show scrollbar (but it's hidden via CSS)
            if (inputRef.current.scrollHeight > 128) {
                inputRef.current.style.overflowY = 'auto';
            } else {
                inputRef.current.style.overflowY = 'hidden';
            }
        }
    }, [inputValue]);

    // Update input value when transcript changes
    useEffect(() => {
        if (transcript) {
            setInputValue(transcript);
        }
    }, [transcript]);

    // Show error toast when voice recognition fails
    useEffect(() => {
        if (voiceError) {
            toast.error(voiceError);
        }
    }, [voiceError]);

    // Handle voice input toggle
    const handleVoiceToggle = () => {
        if (isListening) {
            stopListening();
        } else {
            resetTranscript();
            startListening();
        }
    };

    const handleToggleImportance = async (messageId, isImportant) => {
        if (isImportant) {
            await dispatch(markMessageImportant(messageId));
        } else {
            await dispatch(unmarkMessageImportant(messageId));
        }
        // Always refresh important messages after the operation completes
                if (projectId) {
                    dispatch(getImportantMessages(projectId));
                }
    };

    const handleDeleteMessage = (messageId) => {
        dispatch(deleteMessage(messageId));
    };

    const handleConversationSelect = (conversation) => {
        // This will be handled by the sidebar component
    };

    useEffect(() => {
        // If a new conversation was just created (prev was null, now set), refresh chat history
        if (!prevConversationIdRef.current && conversationId) {
            if (projectId) {
                dispatch(getConversations(projectId));
            }
        }
        prevConversationIdRef.current = conversationId;
    }, [conversationId, projectId, dispatch]);

    const { conversations = [], conversationsStatus } = useSelector((state) => state.chat);

    useEffect(() => {
        if (projectId && conversationsStatus === 'succeeded') {
            if (urlConversationId && !isNewChat) {
                const valid = conversations.some(conv => conv.id === urlConversationId);
                if (valid) {
                    dispatch(getChatHistory(urlConversationId));
                } else if (conversations.length > 0) {
                    window.location.replace(`/user/${projectId}/chat?conversationId=${conversations[0].id}`);
                }
            }
        } else if (projectId && conversationsStatus === 'idle') {
            dispatch(getConversations(projectId));
        }
    }, [projectId, conversationsStatus, urlConversationId, isNewChat, conversations, dispatch]);

    // Helper to check if a message is important (robust string comparison)
    const isMessageImportant = (msgId) => {
        if (!msgId) return false;
        return importantMessages.some(impMsg => String(impMsg.id) === String(msgId));
    };

    // Use messages in their natural chronological order
    const orderedMessages = getMessagesInOrder(messages);

    // Filter out system greetings or bot-initiated messages that are not direct answers
    const filteredMessages = orderedMessages.filter(
        (message) => {
            // Exclude AI messages that are exact system greetings or onboarding
            if (message.role === 'ai' && typeof message.content === 'string') {
                const content = message.content.trim().toLowerCase();
                // Only filter out exact matches, not partials
                const onboardingMessages = [
                    'how can i assist you today?',
                    'welcome to',
                    'i\'m here to help',
                    'what would you like to know?'
                ];
                if (onboardingMessages.some(msg => content === msg)) {
                    return false;
                }
            }
            return true;
        }
    );
    
    // Debug logging for message ordering
    if (messages.length > 0) {
        // console.log('Ordered messages:', orderedMessages.map(m => ({
        //     id: m.id,
        //     role: m.role,
        //     createdAt: m.createdAt || m.created_at
        // })));
    }

    return (
        <div className="w-full h-full flex flex-col">
            <style dangerouslySetInnerHTML={{ __html: modernLLMStyles }} />
            <div className="h-full flex flex-col
                          bg-gray-50 dark:bg-gray-900">
                {/* Modern Chat Header */}
                <div className="px-6 py-4 border-b flex items-center justify-between flex-shrink-0
                              bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <FaRobot className="text-white text-sm" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">{chatHeaderTitle}</h3>
                            <p className="text-xs text-gray-500 dark:text-gray-400">AI Assistant</p>
                        </div>
                    </div>
                </div>

                {/* Chat Messages */}
                <div 
                    ref={chatContainerRef} 
                    className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth relative custom-scrollbar min-h-0"
                    style={{ 
                        scrollbarWidth: 'thin',
                        scrollbarColor: 'rgba(156, 163, 175, 0.4) rgba(0, 0, 0, 0.05)'
                    }}
                >
                    {/* Error message if chat history failed to load */}
                    {status === 'failed' && error && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-red-500 to-pink-500 flex items-center justify-center mx-auto mb-6 shadow-lg">
                                <FaRobot className="text-white text-2xl" />
                            </div>
                            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Conversation Not Found</h2>
                            <p className="text-gray-600 dark:text-gray-400 mb-6">{typeof error === 'string' ? error : 'The conversation you are looking for does not exist or was deleted.'}</p>
                            {conversations && conversations.length > 0 ? (
                                <button
                                    className="bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white rounded-full px-6 py-2 transition-all duration-200 shadow-lg"
                                    onClick={() => window.location.replace(`/user/${projectId}/chat?conversationId=${conversations[0].id}`)}
                                >
                                    Go to First Conversation
                                </button>
                            ) : (
                                <button
                                    className="bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white rounded-full px-6 py-2 transition-all duration-200 shadow-lg"
                                    onClick={() => window.location.replace(`/user/${projectId}/chat?new=1`)}
                                >
                                    Start New Chat
                                </button>
                            )}
                        </div>
                    )}
                    {/* Modern Scroll to bottom button */}
                    {showScrollButton && status !== 'failed' && (
                        <button
                            onClick={scrollToBottom}
                            className="absolute bottom-6 right-6 z-10 p-3 rounded-full bg-white/90 backdrop-blur-sm hover:bg-white text-gray-700 shadow-lg border border-gray-200 transition-all duration-200 hover:scale-110 dark:bg-gray-800/90 dark:text-gray-300 dark:border-gray-600"
                            title="Scroll to bottom"
                        >
                            <FaChevronDown className="w-4 h-4" />
                        </button>
                    )}
                    {messages.length === 0 && !isLoading && status !== 'failed' ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="text-center">
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-4">
                                    <FaRobot className="text-white text-2xl" />
                                </div>
                                <p className="text-gray-500 dark:text-gray-400">Start a conversation with {agentName}</p>
                            </div>
                        </div>
                    ) : status !== 'failed' ? (
                        <>
                            {filteredMessages.map((message, idx) => (
                                <div key={`${conversationId}-${message.id || idx}`} className="flex flex-col">
                                    <MessageBubble 
                                        message={message} 
                                        aiAgentName={agentName} 
                                        onToggleImportance={handleToggleImportance} 
                                        isImportant={isMessageImportant(message.id)} 
                                    />
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex items-start gap-3 max-w-4xl w-full">
                                    <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg">
                                        <FaRobot className="text-white text-sm" />
                                    </div>
                                    <div className="px-4 py-3 rounded-2xl bg-white text-gray-900 shadow-sm border border-gray-200 dark:bg-gray-900 dark:text-gray-100 dark:border-gray-700">
                                        <div className="text-xs font-medium mb-2 opacity-70 text-gray-700 dark:text-gray-300">
                                            {agentName}
                                        </div>
                                        <div className="typing-indicator flex flex-row gap-1.5">
                                            <span className="typing-dot"></span>
                                            <span className="typing-dot" style={{ animationDelay: '0.2s' }}></span>
                                            <span className="typing-dot" style={{ animationDelay: '0.4s' }}></span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    ) : null}
                </div>

                {/* Modern Chat Input */}
                <div className="border-t px-6 py-4 flex-shrink-0
                              bg-white dark:bg-gray-800 border-gray-200
                              dark:border-gray-700">
                    <form onSubmit={handleSubmit} className="flex items-end gap-3">
                        <div className="flex-1 relative">
                            <textarea
                                ref={inputRef}
                                id="userInput"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                placeholder={`Message ${agentName}...`}
                                className="chat-input w-full rounded-2xl py-3 px-4 border outline-none transition resize-none overflow-hidden
                                           bg-white text-gray-900 border-gray-300 placeholder:text-gray-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                                           dark:bg-gray-700 dark:text-white dark:border-gray-600 dark:placeholder:text-gray-400 dark:focus:border-blue-400 dark:focus:ring-blue-400
                                           min-h-[44px] max-h-32"
                                disabled={isLoading}
                                rows={1}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSubmit(e);
                                    }
                                }}
                            />
                            {/* Voice listening indicator */}
                            {isListening && (
                                <div className="absolute top-2 right-2 flex items-center gap-2 px-2 py-1 bg-red-500 text-white rounded-full text-xs animate-pulse">
                                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                                    <span>Listening...</span>
                                </div>
                            )}
                        </div>
                        {isVoiceSupported && (
                            <button
                                type="button"
                                onClick={handleVoiceToggle}
                                className={`rounded-full p-3 transition-all duration-200 shadow-lg ${
                                    isListening
                                        ? 'bg-gradient-to-r from-red-500 to-red-700 hover:from-red-600 hover:to-red-800 text-white animate-pulse'
                                        : 'bg-gradient-to-r from-gray-500 to-gray-700 hover:from-gray-600 hover:to-gray-800 text-white'
                                }`}
                                disabled={isLoading}
                                title={isListening ? 'Stop recording' : 'Start voice input'}
                            >
                                {isListening ? (
                                    <FaMicrophoneSlash className="w-4 h-4" />
                                ) : (
                                    <FaMicrophone className="w-4 h-4" />
                                )}
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={handleGenerateVisualization}
                            className="bg-gradient-to-r from-green-500 to-green-700 hover:from-green-600 hover:to-green-800 text-white rounded-full p-3 transition-all duration-200 disabled:cursor-not-allowed
                                       disabled:bg-gray-300 dark:disabled:bg-gray-600 shadow-lg"
                            disabled={isLoading || !inputValue.trim()}
                            title="Generate Visualization"
                        >
                            <FaChartBar className="w-4 h-4" />
                        </button>
                        <button
                            id="sendButton"
                            type="submit"
                            className="send-button bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white rounded-full p-3 transition-all duration-200 disabled:cursor-not-allowed
                                       disabled:bg-gray-300 dark:disabled:bg-gray-600 shadow-lg"
                            disabled={isLoading || !inputValue.trim()}
                        >
                            <FaPaperPlane className="w-4 h-4" />
                        </button>
                    </form>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
                        {isVoiceSupported 
                            ? 'Press Enter to send, Shift+Enter for new line • Click 🎤 for voice input • Click 📊 for visualizations'
                            : 'Press Enter to send, Shift+Enter for new line • Click 📊 for visualizations'
                        }
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatPage;