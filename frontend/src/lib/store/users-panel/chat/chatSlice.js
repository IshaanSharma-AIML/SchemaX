// src/lib/store/features/chat/chatSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-hot-toast';
import { v4 as uuidv4 } from 'uuid';

const API_URL = process.env.NEXT_PUBLIC_API_BASE;

if (!API_URL) {
    console.error('[CONFIG] NEXT_PUBLIC_API_BASE is not defined. Chat API requests will fail until it is set.');
}

// --- Async Thunk for sending a message to the AI ---
export const sendMessage = createAsyncThunk(
    'chat/sendMessage',
    async (messageData, thunkAPI) => {
        const { naturalLanguageQuery, conversationId, projectId } = messageData;
        const { token } = thunkAPI.getState().auth;
        const state = thunkAPI.getState();
        const wasNewConversation = !state.chat.conversationId;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            console.log('[DEBUG] Sending message to API:', {
                url: `${API_URL}/analyze-data`,
                naturalLanguageQuery,
                conversationId,
                projectId
            });
            
            const response = await fetch(`${API_URL}/analyze-data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({
                    naturalLanguageQuery,
                    conversationId, // Pass the current conversation ID
                    projectId,
                }),
            });
            
            console.log('[DEBUG] API Response status:', response.status);

            const result = await response.json();
            console.log('[DEBUG] API Response data:', result);
            
            const data = result.data;
            if (!data || !data.conversationId) {
                console.log('[DEBUG] Invalid response - missing data or conversationId');
                return thunkAPI.rejectWithValue((data && data.error) || result.error || 'Invalid response from server.');
            }
            // If this was a new conversation, update the title to the first user message
            if (wasNewConversation) {
                thunkAPI.dispatch(updateConversationTitle({ conversationId: data.conversationId, title: naturalLanguageQuery }));
            }
            return data; // Return the full response from the backend
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// --- Async Thunk for generating visualizations ---
export const generateVisualization = createAsyncThunk(
    'chat/generateVisualization',
    async (messageData, thunkAPI) => {
        const { naturalLanguageQuery, conversationId, projectId } = messageData;
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            console.log('[DEBUG] Sending visualization request to API:', {
                url: `${API_URL}/generate-visualization`,
                naturalLanguageQuery,
                conversationId,
                projectId
            });
            
            const response = await fetch(`${API_URL}/generate-visualization`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({
                    naturalLanguageQuery,
                    conversationId,
                    projectId,
                }),
            });
            
            console.log('[DEBUG] Visualization API Response status:', response.status);

            const result = await response.json();
            console.log('[DEBUG] Visualization API Response data:', result);
            
            const data = result.data;
            if (!data) {
                console.log('[DEBUG] Invalid visualization response - missing data');
                return thunkAPI.rejectWithValue((data && data.error) || result.error || 'Invalid response from server.');
            }
            return data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// --- Visualization Management Thunks ---

// Get stored visualizations
export const getVisualizations = createAsyncThunk(
    'chat/getVisualizations',
    async (params, thunkAPI) => {
        const { projectId, conversationId, limit = 50, offset = 0 } = params;
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const queryParams = new URLSearchParams();
            if (projectId) queryParams.append('project_id', projectId);
            if (conversationId) queryParams.append('conversation_id', conversationId);
            queryParams.append('limit', limit);
            queryParams.append('offset', offset);

            const response = await fetch(`${API_URL}/visualizations?${queryParams}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to fetch visualizations');
            }

            const result = await response.json();
            return result.data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Get a specific visualization
export const getVisualization = createAsyncThunk(
    'chat/getVisualization',
    async (visualizationId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/visualizations/${visualizationId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to fetch visualization');
            }

            const result = await response.json();
            return result.data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Toggle favorite status of a visualization
export const toggleFavoriteVisualization = createAsyncThunk(
    'chat/toggleFavoriteVisualization',
    async (visualizationId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/visualizations/${visualizationId}/favorite`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to toggle favorite status');
            }

            const result = await response.json();
            return result.data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Delete a visualization
export const deleteVisualization = createAsyncThunk(
    'chat/deleteVisualization',
    async (visualizationId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/visualizations/${visualizationId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to delete visualization');
            }

            const result = await response.json();
            return { visualizationId, ...result };
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Get chat history for a conversation
export const getChatHistory = createAsyncThunk(
    'chat/getHistory',
    async (conversationId, thunkAPI) => {
        // Defensive check: Only proceed if conversationId is a valid UUID
        if (!conversationId || !/^[0-9a-fA-F-]{36}$/.test(conversationId)) {
            return thunkAPI.rejectWithValue('Invalid or missing conversation ID.');
        }
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to fetch conversation');
            }

            const responseData = await response.json();
            let conversation, messages;
            if (responseData.data && responseData.data.conversation) {
                // Old style: { data: { conversation, messages } }
                conversation = responseData.data.conversation;
                messages = (responseData.data.messages || []).map(msg => ({
                    id: msg.id || msg.ID, // Fallback to 'ID' if 'id' is missing
                    role: msg.role,
                    content: msg.content,
                    queryType: msg.query_type,
                    generatedSql: msg.generated_sql,
                    createdAt: msg.created_at,
                    visualization: msg.visualization // Include visualization data
                }));
            } else if (responseData.id) {
                // New style: conversation object directly
                conversation = responseData;
                messages = []; // No messages returned; you may want to fetch them separately
            } else {
                return thunkAPI.rejectWithValue('Conversation not found.');
            }

            // Messages are already in chronological order from the database

            return {
                conversation,
                messages
            };
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Get all conversations for a project
export const getConversations = createAsyncThunk(
    'chat/getConversations',
    async (projectId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/conversations?project_id=${projectId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            const { data } = await response.json();
            return data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// toggleMessageImportance is deprecated - use markMessageImportant and unmarkMessageImportant instead

// Mark a message as important
export const markMessageImportant = createAsyncThunk(
    'chat/markMessageImportant',
    async (messageId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;
        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }
        try {
            const response = await fetch(`${API_URL}/messages/${messageId}/important`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });
            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to mark as important');
            }
            // Return the message ID for local state update
            return messageId;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Unmark a message as important
export const unmarkMessageImportant = createAsyncThunk(
    'chat/unmarkMessageImportant',
    async (messageId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;
        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }
        try {
            const response = await fetch(`${API_URL}/messages/${messageId}/important`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });
            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to unmark as important');
            }
            // Return the message ID for local state update
            return messageId;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Get important messages
export const getImportantMessages = createAsyncThunk(
    'chat/getImportantMessages',
    async (projectId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const url = projectId 
                ? `${API_URL}/important-messages?project_id=${projectId}`
                : `${API_URL}/important-messages`;
                
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            const { data } = await response.json();
            return data;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Delete conversation
export const deleteConversation = createAsyncThunk(
    'chat/deleteConversation',
    async (conversationId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to delete conversation');
            }

            return conversationId;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Delete message
export const deleteMessage = createAsyncThunk(
    'chat/deleteMessage',
    async (messageId, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;

        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }

        try {
            const response = await fetch(`${API_URL}/messages/${messageId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to delete message');
            }

            return messageId;
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Update conversation title
export const updateConversationTitle = createAsyncThunk(
    'chat/updateConversationTitle',
    async ({ conversationId, title }, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;
        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }
        try {
            const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({ title }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to update conversation title');
            }
            return { conversationId, title };
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

// Rename (update) conversation title
export const updateConversation = createAsyncThunk(
    'chat/updateConversation',
    async ({ conversationId, title }, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;
        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }
        try {
            const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({ title }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to update conversation');
            }
            return { conversationId, title };
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

export const updateImportantMessageTitle = createAsyncThunk(
    'chat/updateImportantMessageTitle',
    async ({ messageId, title }, thunkAPI) => {
        const { token } = thunkAPI.getState().auth;
        if (!token) {
            return thunkAPI.rejectWithValue('No authorization token found. Please log in.');
        }
        try {
            const response = await fetch(`${API_URL}/messages/${messageId}/important-title`, {
                method: 'PATCH',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ title }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                return thunkAPI.rejectWithValue(errorData.detail || 'Failed to update title');
            }
            return { messageId, title };
        } catch (error) {
            const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
            return thunkAPI.rejectWithValue(message);
        }
    }
);

const initialState = {
    messages: [],           // Stores the list of messages in the current chat
    conversationId: null,   // Stores the ID of the current conversation
    conversations: [],      // Stores all conversations for the current project
    importantMessages: [],  // Stores important messages
    currentConversation: null, // Current conversation details
    visualizations: [],      // Stores all visualizations for the current project/conversation
    currentVisualization: null, // Current visualization details
    status: 'idle',        // 'idle' | 'loading' | 'succeeded' | 'failed' - for AI responses
    conversationsStatus: 'idle', // Separate status for conversations
    importantMessagesStatus: 'idle', // Separate status for important messages
    importanceOperationStatus: 'idle', // Separate status for importance operations
    visualizationsStatus: 'idle', // Separate status for visualizations
    error: null,
};

export const chatSlice = createSlice({
    name: 'chat',
    initialState,
    reducers: {
        // This standard reducer instantly adds the user's message to the UI
        addHumanMessage: (state, action) => {
            state.messages.push({
                role: 'human',
                content: action.payload.content,
                id: action.payload.id || null, // Don't generate ID here, will be set by backend
                createdAt: action.payload.createdAt || new Date().toISOString(),
            });
            // If there's no conversation ID yet, we can't get a response, but we show the message
            if (!state.conversationId) {
                state.conversationId = action.payload.conversationId;
            }
        },
        addAiMessage: (state, action) => {
            state.messages.push({
                role: 'ai',
                content: action.payload.content,
                id: action.payload.id || null,
                createdAt: action.payload.createdAt || new Date().toISOString(),
                queryType: action.payload.queryType || null,
                generatedSql: action.payload.generatedSql || null,
                isImportant: false,
                visualization: action.payload.visualization || null
            });
        },
        // Used to clear the chat when a user navigates away or starts a new project chat
        clearChat: (state) => {
            state.messages = [];
            state.conversationId = null;
            state.currentConversation = null;
            state.status = 'idle';
            state.error = null;
        },
        
        // Set current conversation
        setCurrentConversation: (state, action) => {
            state.currentConversation = action.payload;
        },
        
        // Update message importance locally
        updateMessageImportance: (state, action) => {
            const { messageId, isImportant } = action.payload;
            const message = state.messages.find(msg => msg.id === messageId);
            if (message) {
                message.isImportant = isImportant;
            }
        },
    },
    extraReducers: (builder) => {
        builder
            // sendMessage Thunk
            .addCase(sendMessage.pending, (state) => {
                state.status = 'loading'; // Used to show "AI is typing..."
            })
            .addCase(sendMessage.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.conversationId = action.payload.conversationId; // Update the conversationId

                // Find the last user message without an ID (optimistic message)
                const lastUserMessageIndex = state.messages.findIndex(msg => msg.role === 'human' && !msg.id);
                let aiMessage = {
                    role: 'ai',
                    content: action.payload.analysis, // The AI's response text
                    queryType: action.payload.queryType,
                    generatedSql: action.payload.generatedSql,
                    id: action.payload.aiMessageId || uuidv4(),
                    createdAt: action.payload.aiCreatedAt || new Date().toISOString(),
                };
                
                // Add visualization data if present
                if (action.payload.visualization) {
                    aiMessage.visualization = action.payload.visualization;
                }
                
                if (lastUserMessageIndex !== -1 && action.payload.userMessageId) {
                    // Update the user message with the correct ID and createdAt from backend
                    state.messages[lastUserMessageIndex].id = action.payload.userMessageId;
                    if (action.payload.userCreatedAt) {
                        state.messages[lastUserMessageIndex].createdAt = action.payload.userCreatedAt;
                    }
                    // Insert the AI message immediately after the user message
                    state.messages.splice(lastUserMessageIndex + 1, 0, aiMessage);
                } else {
                    // Fallback: just push the AI message
                    state.messages.push(aiMessage);
                }
            })
            .addCase(sendMessage.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
                state.messages.push({
                    role: 'ai',
                    content: `Sorry, an error occurred: ${action.payload}`,
                    isError: true, // Add a flag for styling error messages
                });
                toast.error("An error occurred while getting a response.");
            })
            // getChatHistory Thunk
            .addCase(getChatHistory.fulfilled, (state, action) => {
                state.messages = action.payload.messages; // Replace messages with fetched history
                state.currentConversation = action.payload.conversation;
                state.conversationId = action.payload.conversation.id;
                state.status = 'succeeded';
            })
            .addCase(getChatHistory.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
            })
            // getConversations Thunk
            .addCase(getConversations.pending, (state) => {
                state.conversationsStatus = 'loading';
            })
            .addCase(getConversations.fulfilled, (state, action) => {
                state.conversations = action.payload;
                state.conversationsStatus = 'succeeded';
            })
            .addCase(getConversations.rejected, (state, action) => {
                state.conversationsStatus = 'failed';
                state.error = action.payload;
            })
            // toggleMessageImportance is deprecated - use markMessageImportant and unmarkMessageImportant instead
            // getImportantMessages Thunk
            .addCase(getImportantMessages.pending, (state) => {
                state.importantMessagesStatus = 'loading';
            })
            .addCase(getImportantMessages.fulfilled, (state, action) => {
                state.importantMessages = action.payload;
                state.importantMessagesStatus = 'succeeded';
            })
            .addCase(getImportantMessages.rejected, (state, action) => {
                state.importantMessagesStatus = 'failed';
                state.error = action.payload;
            })
            // deleteConversation Thunk
            .addCase(deleteConversation.pending, (state) => {
                state.status = 'loading';
            })
            .addCase(deleteConversation.fulfilled, (state, action) => {
                // Remove the deleted conversation from the list
                state.conversations = state.conversations.filter(
                    conv => conv.id !== action.payload
                );
                // If the deleted conversation was the current one, clear it
                if (state.conversationId === action.payload) {
                    state.conversationId = null;
                    state.currentConversation = null;
                    state.messages = [];
                }
                state.status = 'succeeded';
                toast.success('Conversation deleted successfully');
            })
            .addCase(deleteConversation.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
                toast.error(action.payload || 'Failed to delete conversation');
            })
            // deleteMessage Thunk
            .addCase(deleteMessage.pending, (state) => {
                state.status = 'loading';
            })
            .addCase(deleteMessage.fulfilled, (state, action) => {
                // Remove the deleted message from the current messages
                state.messages = state.messages.filter(msg => msg.id !== action.payload);
                // Remove from important messages if it was there
                state.importantMessages = state.importantMessages.filter(msg => msg.id !== action.payload);
                state.status = 'succeeded';
                toast.success('Message deleted successfully');
            })
            .addCase(deleteMessage.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
                toast.error(action.payload || 'Failed to delete message');
            })
            // updateConversation Thunk
            .addCase(updateConversation.fulfilled, (state, action) => {
                const { conversationId, title } = action.payload;
                const conv = state.conversations.find(c => c.id === conversationId);
                if (conv) conv.title = title;
            })
            // markMessageImportant Thunk
            .addCase(markMessageImportant.pending, (state) => {
                state.importanceOperationStatus = 'loading';
            })
            .addCase(markMessageImportant.fulfilled, (state, action) => {
                state.importanceOperationStatus = 'succeeded';
                // Refresh important messages to get the updated list
                // This will be handled by the component that calls markMessageImportant
            })
            .addCase(markMessageImportant.rejected, (state, action) => {
                state.importanceOperationStatus = 'failed';
                state.error = action.payload;
                toast.error(action.payload || 'Failed to mark message as important');
            })
            // unmarkMessageImportant Thunk
            .addCase(unmarkMessageImportant.pending, (state) => {
                state.importanceOperationStatus = 'loading';
            })
            .addCase(unmarkMessageImportant.fulfilled, (state, action) => {
                state.importanceOperationStatus = 'succeeded';
                // Refresh important messages to get the updated list
                // This will be handled by the component that calls unmarkMessageImportant
            })
            .addCase(unmarkMessageImportant.rejected, (state, action) => {
                state.importanceOperationStatus = 'failed';
                state.error = action.payload;
                toast.error(action.payload || 'Failed to unmark message as important');
            })
            .addCase(updateImportantMessageTitle.fulfilled, (state, action) => {
                const { messageId, title } = action.payload;
                const msg = state.importantMessages.find(m => m.id === messageId);
                if (msg) msg.title = title;
            })
            .addCase(updateImportantMessageTitle.rejected, (state, action) => {
                toast.error(action.payload || 'Failed to update important message title');
            })
            // generateVisualization Thunk
            .addCase(generateVisualization.pending, (state) => {
                state.status = 'loading';
            })
            .addCase(generateVisualization.fulfilled, (state, action) => {
                state.status = 'succeeded';
                console.log('[DEBUG] generateVisualization.fulfilled - action.payload:', action.payload);
                console.log('[DEBUG] generateVisualization.fulfilled - current messages:', state.messages);
                console.log('[DEBUG] generateVisualization.fulfilled - messages count:', state.messages.length);
                state.messages.forEach((msg, index) => {
                    console.log(`[DEBUG] Message ${index}: role=${msg.role}, content="${msg.content?.substring(0, 50)}...", hasVisualization=${!!msg.visualization}`);
                });
                
                // Extract visualization data from the response
                const visualizationData = action.payload.visualization;
                console.log('[DEBUG] generateVisualization.fulfilled - visualizationData:', visualizationData);
                
                // Find the last AI message and add visualization to it
                let lastAiMessageIndex = -1;
                for (let i = state.messages.length - 1; i >= 0; i--) {
                    if (state.messages[i].role === 'ai') {
                        lastAiMessageIndex = i;
                        break;
                    }
                }
                console.log('[DEBUG] generateVisualization.fulfilled - lastAiMessageIndex:', lastAiMessageIndex);
                
                if (lastAiMessageIndex !== -1 && visualizationData) {
                    console.log('[DEBUG] generateVisualization.fulfilled - adding visualization to message at index:', lastAiMessageIndex);
                    state.messages[lastAiMessageIndex].visualization = visualizationData;
                    console.log('[DEBUG] generateVisualization.fulfilled - updated message:', state.messages[lastAiMessageIndex]);
                } else {
                    console.log('[DEBUG] generateVisualization.fulfilled - No AI message found or no visualization data');
                }
            })
            .addCase(generateVisualization.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
                toast.error("Failed to generate visualization");
            })
            // getVisualizations Thunk
            .addCase(getVisualizations.pending, (state) => {
                state.visualizationsStatus = 'loading';
            })
            .addCase(getVisualizations.fulfilled, (state, action) => {
                state.visualizations = action.payload;
                state.visualizationsStatus = 'succeeded';
            })
            .addCase(getVisualizations.rejected, (state, action) => {
                state.visualizationsStatus = 'failed';
                state.error = action.payload;
                toast.error(action.payload || 'Failed to fetch visualizations');
            })
            // getVisualization Thunk
            .addCase(getVisualization.fulfilled, (state, action) => {
                state.currentVisualization = action.payload;
            })
            .addCase(getVisualization.rejected, (state, action) => {
                state.error = action.payload;
                toast.error(action.payload || 'Failed to fetch visualization');
            })
            // toggleFavoriteVisualization Thunk
            .addCase(toggleFavoriteVisualization.fulfilled, (state, action) => {
                const { visualizationId, isFavorite } = action.payload;
                const visualization = state.visualizations.find(v => v.id === visualizationId);
                if (visualization) {
                    visualization.is_favorite = isFavorite;
                }
                if (state.currentVisualization && state.currentVisualization.id === visualizationId) {
                    state.currentVisualization.is_favorite = isFavorite;
                }
                toast.success(isFavorite ? 'Added to favorites' : 'Removed from favorites');
            })
            .addCase(toggleFavoriteVisualization.rejected, (state, action) => {
                state.error = action.payload;
                toast.error(action.payload || 'Failed to update favorite status');
            })
            // deleteVisualization Thunk
            .addCase(deleteVisualization.fulfilled, (state, action) => {
                const { visualizationId } = action.payload;
                state.visualizations = state.visualizations.filter(v => v.id !== visualizationId);
                if (state.currentVisualization && state.currentVisualization.id === visualizationId) {
                    state.currentVisualization = null;
                }
                toast.success('Visualization deleted successfully');
            })
            .addCase(deleteVisualization.rejected, (state, action) => {
                state.error = action.payload;
                toast.error(action.payload || 'Failed to delete visualization');
            });
    },
});

export const { addHumanMessage, addAiMessage, clearChat, setCurrentConversation, updateMessageImportance } = chatSlice.actions;
export default chatSlice.reducer;