'use client';

import { useState, useRef, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useParams, usePathname, useSearchParams, useRouter } from 'next/navigation';
import { 
    FaComments, 
    FaChevronRight,
    FaChevronDown,
    FaTrash,
    FaEdit,
    FaEllipsisV,
    FaPlus,
    FaStar
} from 'react-icons/fa';
import { 
    getConversations, 
    deleteConversation, 
    updateConversation,
    getImportantMessages
} from '@/lib/store/users-panel/chat/chatSlice';
import { format, isValid, isToday, isYesterday } from 'date-fns';

// Enhanced CSS for sidebar scrollbar
const sidebarScrollbarStyles = `
  /* Modern scrollbar for sidebar - always visible */
  .sidebar-scrollbar {
    scrollbar-width: thin !important;
    scrollbar-color: rgba(156, 163, 175, 0.6) rgba(0, 0, 0, 0.1) !important;
  }
  
  .sidebar-scrollbar::-webkit-scrollbar {
    width: 10px !important;
    -webkit-appearance: none;
  }
  
  .sidebar-scrollbar::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
    border-radius: 5px;
    -webkit-box-shadow: inset 0 0 1px rgba(0,0,0,0.1);
  }
  
  .sidebar-scrollbar::-webkit-scrollbar-thumb {
    background: rgba(156, 163, 175, 0.6) !important;
    border-radius: 5px;
    transition: background 0.2s ease;
    -webkit-box-shadow: inset 0 0 1px rgba(0,0,0,0.1);
  }
  
  .sidebar-scrollbar::-webkit-scrollbar-thumb:hover {
    background: rgba(156, 163, 175, 0.8) !important;
  }
  
  .dark .sidebar-scrollbar {
    scrollbar-color: rgba(75, 85, 99, 0.4) rgba(255, 255, 255, 0.05) !important;
  }
  
  .dark .sidebar-scrollbar::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05) !important;
  }
  
  .dark .sidebar-scrollbar::-webkit-scrollbar-thumb {
    background: rgba(75, 85, 99, 0.4) !important;
  }
  
  .dark .sidebar-scrollbar::-webkit-scrollbar-thumb:hover {
    background: rgba(75, 85, 99, 0.6) !important;
  }
`;

const Sidebar = () => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [openMenuId, setOpenMenuId] = useState(null);
    const [editingId, setEditingId] = useState(null);
    const [editTitle, setEditTitle] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [conversationToDelete, setConversationToDelete] = useState(null);
    const menuRef = useRef(null);
    const editInputRef = useRef(null);

    const dispatch = useDispatch();
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const params = useParams();
    const projectId = params?.projectId;
    const currentConversationId = searchParams.get('conversationId');

    const { 
        conversations = [], 
        conversationsStatus,
        importantMessages = [],
        importantMessagesStatus 
    } = useSelector((state) => state.chat);

    // Load conversations when component mounts
    useEffect(() => {
        if (projectId && conversationsStatus === 'idle') {
            dispatch(getConversations(projectId));
        }
    }, [projectId, conversationsStatus, dispatch]);

    // Load important messages
    useEffect(() => {
        if (projectId) {
            dispatch(getImportantMessages(projectId));
        }
    }, [projectId, dispatch]);

    // Handle click outside to close menus
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setOpenMenuId(null);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    // Focus edit input when editing starts
    useEffect(() => {
        if (editingId && editInputRef.current) {
            editInputRef.current.focus();
            editInputRef.current.select();
        }
    }, [editingId]);

    const handleMenuToggle = (e, conversationId) => {
        e.stopPropagation();
        setOpenMenuId(prevId => (prevId === conversationId ? null : conversationId));
    };

    const handleConversationClick = (conversationId) => {
        router.push(`/user/${projectId}/chat?conversationId=${conversationId}`);
    };

    const handleNewChat = () => {
        console.log('New Chat button clicked, projectId:', projectId);
        if (projectId) {
            // Use replace to avoid adding to history and ensure clean state
            // Add timestamp to force navigation even if already on chat page
            const timestamp = Date.now();
            router.replace(`/user/${projectId}/chat?new=1&t=${timestamp}`);
        } else {
            console.error('No projectId available for new chat');
        }
    };

    const handleEditStart = (e, conversation) => {
        e.stopPropagation();
        setEditingId(conversation.id);
        setEditTitle(conversation.title || '');
        setOpenMenuId(null);
    };

    const handleEditSave = async (e) => {
        e.stopPropagation();
        if (editTitle.trim() && editingId) {
            await dispatch(updateConversation({ 
                conversationId: editingId, 
                title: editTitle.trim() 
            }));
            setEditingId(null);
            setEditTitle('');
        }
    };

    const handleEditCancel = (e) => {
        e.stopPropagation();
        setEditingId(null);
        setEditTitle('');
    };

    const handleEditKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleEditSave(e);
        } else if (e.key === 'Escape') {
            handleEditCancel(e);
        }
    };

    const handleDeleteClick = (e, conversation) => {
        e.stopPropagation();
        setConversationToDelete(conversation);
        setIsModalOpen(true);
        setOpenMenuId(null);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setConversationToDelete(null);
    };

    const handleConfirmDelete = async () => {
        if (conversationToDelete) {
            await dispatch(deleteConversation(conversationToDelete.id));
            // If we deleted the current conversation, redirect to new chat
            if (currentConversationId === conversationToDelete.id) {
                router.push(`/user/${projectId}/chat?new=1`);
            }
        }
        handleCloseModal();
    };

    const handleMarkImportant = async (e, conversation) => {
        e.stopPropagation();
        // For conversations, we'll mark the first message as important
        // This is a simplified approach - you might want to implement conversation-level importance
        setOpenMenuId(null);
        // Note: This would need backend support for conversation-level importance
    };

    const formatDate = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        if (!isValid(date)) return '';
        return format(date, 'MMM d, h:mm a');
    };

    const formatDateHeader = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        if (!isValid(date)) return '';
        
        // Use date-fns helpers for better timezone handling
        if (isToday(date)) {
            return 'Today';
        } else if (isYesterday(date)) {
            return 'Yesterday';
        } else {
            return format(date, 'MMM d, yyyy');
        }
    };

    const groupConversationsByDate = (conversations) => {
        if (!conversations || conversations.length === 0) return {};
        
        const sortedConversations = [...conversations].sort((a, b) => {
            const dateA = new Date(a.created_at);
            const dateB = new Date(b.created_at);
            return dateB - dateA; // Most recent first
        });

        const grouped = {};
        sortedConversations.forEach(conversation => {
            const dateHeader = formatDateHeader(conversation.created_at);
            if (!grouped[dateHeader]) {
                grouped[dateHeader] = [];
            }
            grouped[dateHeader].push(conversation);
        });

        return grouped;
    };

    const isConversationImportant = (conversationId) => {
        // Check if any messages in this conversation are marked as important
        return importantMessages.some(msg => msg.conversation_id === conversationId);
    };

    // Navigation items removed - keeping only chat history

    return (
        <aside className={`${isExpanded ? 'w-80' : 'w-16'} transition-all duration-300 flex-shrink-0`}>
            <style dangerouslySetInnerHTML={{ __html: sidebarScrollbarStyles }} />
            <div className="h-full bg-white dark:bg-gray-900/50 border-r border-gray-200 dark:border-gray-800/30 flex flex-col">
                {/* Header */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-800/30">
                    <div className="flex items-center justify-between" onClick={(e) => e.stopPropagation()}>
                        {isExpanded && (
                            <div>
                                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                                    Chat History
                                </h2>
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                    {conversations.length} conversations
                                </p>
                            </div>
                        )}
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                setIsExpanded(!isExpanded);
                            }}
                            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/40 transition-colors"
                        >
                            {isExpanded ? (
                                <FaChevronDown className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                            ) : (
                                <FaChevronRight className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                            )}
                        </button>
                    </div>
                </div>

                {/* Chat History Section */}
                {isExpanded && (
                    <div className="flex-1 flex flex-col">

                        {/* New Chat Button */}
                        <div className="p-4 border-b border-gray-200 dark:border-gray-800/30">
                            <button
                                onClick={(e) => {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    console.log('Button clicked');
                                    handleNewChat();
                                }}
                                disabled={!projectId}
                                className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                                    projectId 
                                        ? 'bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white cursor-pointer' 
                                        : 'bg-gray-400 text-gray-200 cursor-not-allowed'
                                }`}
                            >
                                <FaPlus className="w-4 h-4" />
                                New Chat
                            </button>
                        </div>

                        {/* Conversations List */}
                        <div className="flex-1 sidebar-scrollbar" style={{ 
                            scrollbarWidth: 'thin',
                            scrollbarColor: 'rgba(156, 163, 175, 0.6) rgba(0, 0, 0, 0.1)',
                            overflowY: 'scroll',
                            maxHeight: 'calc(100vh - 200px)',
                            minHeight: '400px'
                        }}>
                            {conversationsStatus === 'loading' ? (
                                <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                                    Loading conversations...
                                </div>
                            ) : conversations.length === 0 ? (
                                <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                                    No conversations yet
                                </div>
                            ) : (
                                <div className="p-2">
                                    {Object.entries(groupConversationsByDate(conversations)).map(([dateHeader, dateConversations]) => (
                                        <div key={dateHeader} className="mb-4">
                                            {/* Date Header */}
                                            <div className="px-3 py-2 mb-2">
                                                <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                                                    {dateHeader}
                                                </h4>
                                            </div>
                                            
                                            {/* Conversations for this date */}
                                            <div className="space-y-1">
                                                {dateConversations.map((conversation) => {
                                                    const isActive = currentConversationId === conversation.id;
                                                    const isImportant = isConversationImportant(conversation.id);
                                                    
                                                    return (
                                                        <div
                                                            key={conversation.id}
                                                            className={`group relative rounded-lg transition-all duration-200 ${
                                                                isActive
                                                                    ? 'bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800/30'
                                                                    : 'hover:bg-gray-50 dark:hover:bg-gray-800/30'
                                                            }`}
                                                        >
                                                            <div
                                                                className="flex items-center gap-3 p-3 cursor-pointer"
                                                                onClick={() => handleConversationClick(conversation.id)}
                                                            >
                                                                <FaComments className={`w-4 h-4 flex-shrink-0 ${
                                                                    isActive ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400'
                                }`} />
                                                                
                                    <div className="flex-1 min-w-0">
                                                                    {editingId === conversation.id ? (
                                                                        <input
                                                                            ref={editInputRef}
                                                                            type="text"
                                                                            value={editTitle}
                                                                            onChange={(e) => setEditTitle(e.target.value)}
                                                                            onKeyDown={handleEditKeyPress}
                                                                            onBlur={handleEditSave}
                                                                            className="w-full text-sm font-medium bg-transparent border-none outline-none text-gray-900 dark:text-white"
                                                                            onClick={(e) => e.stopPropagation()}
                                                                        />
                                                                    ) : (
                                                                        <div className="flex items-center gap-2">
                                                                            <h3 className={`text-sm font-medium truncate ${
                                                                                isActive ? 'text-blue-900 dark:text-blue-100' : 'text-gray-900 dark:text-white'
                                                                            }`}>
                                                                                {conversation.title || 'Untitled Chat'}
                                                                            </h3>
                                                                            {isImportant && (
                                                                                <FaStar className="w-3 h-3 text-yellow-500 flex-shrink-0" />
                                                                            )}
                                                                        </div>
                                                                    )}
                                                                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                                                        {formatDate(conversation.created_at)}
                                                                    </p>
                                                                </div>
                                                                
                                                                <div className="flex items-center gap-1">
                                                                    <button
                                                                        onClick={(e) => handleMenuToggle(e, conversation.id)}
                                                                        className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700/50 opacity-0 group-hover:opacity-100 transition-opacity"
                                                                    >
                                                                        <FaEllipsisV className="w-3 h-3 text-gray-500" />
                                                                    </button>
                                                                    
                                                                    {/* Dropdown Menu */}
                                                                    {openMenuId === conversation.id && (
                                                                        <div
                                                                            ref={menuRef}
                                                                            className="absolute right-2 top-full mt-1 w-48 bg-white dark:bg-gray-900/80 rounded-lg shadow-lg border border-gray-200 dark:border-gray-800/30 z-50"
                                                                        >
                                                                            <ul className="py-1">
                                                                                <li>
                                                                                    <button
                                                                                        onClick={(e) => handleEditStart(e, conversation)}
                                                                                        className="w-full text-left px-4 py-2 text-sm flex items-center gap-2 text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/40"
                                                                                    >
                                                                                        <FaEdit className="w-3 h-3" />
                                                                                        Rename
                                                                                    </button>
                                                                                </li>
                                                                                <li>
                                                                                    <button
                                                                                        onClick={(e) => handleMarkImportant(e, conversation)}
                                                                                        className="w-full text-left px-4 py-2 text-sm flex items-center gap-2 text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800/40"
                                                                                    >
                                                                                        <FaStar className="w-3 h-3" />
                                                                                        {isImportant ? 'Remove from Important' : 'Mark as Important'}
                                                                                    </button>
                                                                                </li>
                                                                                <li>
                                                                                    <button
                                                                                        onClick={(e) => handleDeleteClick(e, conversation)}
                                                                                        className="w-full text-left px-4 py-2 text-sm flex items-center gap-2 text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-gray-800/40"
                                                                                    >
                                                                                        <FaTrash className="w-3 h-3" />
                                                                                        Delete
                                                                                    </button>
                                                                                </li>
                                                                            </ul>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                                        </div>
                                    </div>
                                )}

                {/* Footer */}
                {isExpanded && (
                    <div className="p-4 border-t border-gray-200 dark:border-gray-800/30">
                        <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
                            SchemaX AI Assistant
                        </div>
                    </div>
                )}
            </div>

            {/* Delete Confirmation Modal */}
            {isModalOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white dark:bg-gray-900/80 rounded-lg p-6 max-w-md w-full mx-4">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                            Delete Conversation
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-6">
                            Are you sure you want to delete "{conversationToDelete?.title || 'Untitled Chat'}"? 
                            This action cannot be undone.
                        </p>
                        <div className="flex gap-3 justify-end">
                            <button
                                onClick={handleCloseModal}
                                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleConfirmDelete}
                                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </aside>
    );
};

export default Sidebar;
