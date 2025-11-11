import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-hot-toast';
import Cookies from 'js-cookie';

const API_URL = process.env.NEXT_PUBLIC_API_BASE;

if (!API_URL) {
    console.error('[CONFIG] NEXT_PUBLIC_API_BASE is not defined. Auth API requests will fail until it is set.');
}

// Safely get stored user data
const getStoredUser = () => {
    if (typeof window !== 'undefined') {
        try {
            const user = localStorage.getItem('user');
            return user ? JSON.parse(user) : null;
        } catch (error) {
            console.error('Error parsing stored user:', error);
            return null;
        }
    }
    return null;
};

const getStoredToken = () => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('token');
    }
    return null;
};

const storedUser = getStoredUser();
const storedToken = getStoredToken();

// Helper function to get specific error message
const getErrorMessage = (error, response) => {
    if (response?.status === 409) {
        return "A user with this email already exists.";
    } else if (response?.status === 401) {
        return "Invalid email or password.";
    } else if (response?.status === 500) {
        return "Server error. Please try again later.";
    } else if (response?.status === 0 || !response) {
        return "Unable to connect to server. Please check your internet connection.";
    } else if (error?.message) {
        return error.message;
    } else {
        return "An unexpected error occurred. Please try again.";
    }
};

// --- Thunks (Now cleaner) ---

export const registerUser = createAsyncThunk(
    'auth/register',
    async (userData, thunkAPI) => {
        try {
            const response = await fetch(`${API_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                const errorMessage = getErrorMessage(data, response);
                return thunkAPI.rejectWithValue({ message: errorMessage, status: response.status });
            }
            
            if (!data.success) {
                return thunkAPI.rejectWithValue({ message: data.message || "Registration failed", status: response.status });
            }
            
            // Return the data directly (not data.data) for Python backend
            return { user: data.user, token: data.token };
        } catch (error) {
            const errorMessage = getErrorMessage(error, null);
            return thunkAPI.rejectWithValue({ message: errorMessage, status: 0 });
        }
    }
);

export const loginUser = createAsyncThunk(
    'auth/login',
    async (userData, thunkAPI) => {
        try {
            const response = await fetch(`${API_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                const errorMessage = getErrorMessage(data, response);
                return thunkAPI.rejectWithValue({ message: errorMessage, status: response.status });
            }
            
            if (!data.success) {
                return thunkAPI.rejectWithValue({ message: data.message || "Login failed", status: response.status });
            }
            
            // Return the data directly (not data.data) for Python backend
            return { user: data.user, token: data.token };
        } catch (error) {
            const errorMessage = getErrorMessage(error, null);
            return thunkAPI.rejectWithValue({ message: errorMessage, status: 0 });
        }
    }
);

// --- Auth Slice Definition ---

const initialState = {
    user: storedUser || null,
    token: storedToken || null,
    isLoading: false,
    isSuccess: false,
    isError: false,
    message: '',
    errorType: '', // New field to track error type
};

export const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        reset: (state) => {
            state.isLoading = false;
            state.isSuccess = false;
            state.isError = false;
            state.message = '';
            state.errorType = '';
        },
        logout: (state) => {
            Cookies.remove('authToken');
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            state.user = null;
            state.token = null;
            toast.success("You have been logged out.");
            if (typeof window !== 'undefined') {
                window.location.href = '/login';
            }
        }
    },
    extraReducers: (builder) => {
        builder
            // Register User cases
            .addCase(registerUser.pending, (state) => {
                state.isLoading = true;
                state.isError = false;
                state.message = '';
                state.errorType = '';
            })
            .addCase(registerUser.fulfilled, (state, action) => {
                const { user, token } = action.payload;

                state.isLoading = false;
                state.isSuccess = true;
                state.user = user;
                state.token = token;

                localStorage.setItem('user', JSON.stringify(user));
                localStorage.setItem('token', token);
                Cookies.set('authToken', token, {
                    expires: 1,
                    secure: process.env.NODE_ENV === 'production',
                    sameSite: 'strict'
                });

                toast.success(`Welcome, ${user.name}!`);
            })
            .addCase(registerUser.rejected, (state, action) => {
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload?.message || 'Registration failed';
                state.errorType = action.payload?.status === 409 ? 'user_exists' : 
                                 action.payload?.status === 0 ? 'server_offline' : 'other';
                // Don't show toast here - let the UI handle the error display
            })
            
            // Login User cases
            .addCase(loginUser.pending, (state) => {
                state.isLoading = true;
                state.isError = false;
                state.message = '';
                state.errorType = '';
            })
            .addCase(loginUser.fulfilled, (state, action) => {
                const { user, token } = action.payload;

                // --- THIS IS THE CORRECTED LOGIC ---
                // 1. Update the state
                state.isLoading = false;
                state.isSuccess = true;
                state.user = user;
                state.token = token;

                // 2. Perform all storage side effects here, in one place
                localStorage.setItem('user', JSON.stringify(user));
                localStorage.setItem('token', token);
                Cookies.set('authToken', token, {
                    expires: 1,
                    secure: process.env.NODE_ENV === 'production',
                    sameSite: 'strict'
                });

                toast.success(`Welcome back, ${user.name}!`);
            })
            .addCase(loginUser.rejected, (state, action) => {
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload?.message || 'Login failed';
                state.errorType = action.payload?.status === 401 ? 'invalid_credentials' : 
                                 action.payload?.status === 0 ? 'server_offline' : 'other';
                // Don't show toast here - let the UI handle the error display
            });
    },
});

export const { reset, logout } = authSlice.actions;
export default authSlice.reducer;