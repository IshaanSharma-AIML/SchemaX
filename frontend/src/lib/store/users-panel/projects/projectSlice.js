// src/lib/store/features/projects/projectSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-hot-toast';

const API_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000/api';

if (!process.env.NEXT_PUBLIC_API_BASE) {
    console.warn('[CONFIG] NEXT_PUBLIC_API_BASE not set; defaulting to http://localhost:8000/api for local development.');
}

// --- Reusable function to handle API requests ---
const makeApiRequest = async (url, method, body, thunkAPI) => {
    const { token } = thunkAPI.getState().auth;
    
    // Check if this is a demo token
    if (token && token.startsWith('demo-token-')) {
        // Return demo data for demo tokens
        if (method === 'GET' && url.includes('/projects')) {
            return []; // Empty projects array for demo
        }
        return null;
    }
    
    try {
        const response = await fetch(url, {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            ...(body && { body: JSON.stringify(body) }),
        });
        const data = await response.json();
        if (!data.success) {
            return thunkAPI.rejectWithValue(data.message);
        }
        // Fix for single project fetch: return data.project
        if (url.match(/\/projects\/[^/]+$/) && method === 'GET') {
            return data.project;
        }
        // Python backend returns data.projects for list endpoints
        return data.projects;
    } catch (error) {
        const message = (error.response && error.response.data && error.response.data.message) || error.message || error.toString();
        return thunkAPI.rejectWithValue(message);
    }
};

// --- Async Thunks for Project CRUD Operations ---

// CREATE a new project
export const createProject = createAsyncThunk(
    'projects/create',
    (projectData, thunkAPI) => makeApiRequest(`${API_URL}/projects`, 'POST', projectData, thunkAPI)
);

// LIST all projects for the user
export const getProjects = createAsyncThunk(
    'projects/getAll',
    (_, thunkAPI) => makeApiRequest(`${API_URL}/projects`, 'GET', null, thunkAPI)
);

// GET a single project by its ID
export const getProjectById = createAsyncThunk(
    'projects/getOne',
    (projectId, thunkAPI) => makeApiRequest(`${API_URL}/projects/${projectId}`, 'GET', null, thunkAPI)
);

// DELETE a project by its ID
export const deleteProject = createAsyncThunk(
    'projects/delete',
    (projectId, thunkAPI) => makeApiRequest(`${API_URL}/projects/${projectId}`, 'DELETE', null, thunkAPI)
);

// UPDATE a project
export const updateProject = createAsyncThunk(
    'projects/update',
    ({ projectId, projectData }, thunkAPI) =>
        makeApiRequest(`${API_URL}/projects/${projectId}`, 'PUT', projectData, thunkAPI)
);


// --- Project Slice Definition ---

const initialState = {
    projects: [], // Array to hold the list of all user projects
    project: {
        projectName: '',
        projectInfo: '',
        dbHost: '',
        dbUser: '',
        dbPassword: '',
        databaseName: '',
        dbInfo: '',
        botName: '',
        botAvatar: '',
    },  // Object to hold details of a single project when viewed
    isLoading: false,
    isSuccess: false,
    isError: false,
    message: '',
    status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
};

export const projectSlice = createSlice({
    name: 'projects',
    initialState,
    reducers: {
        reset: (state) => {
            state.isLoading = false;
            state.isSuccess = false;
            state.isError = false;
            state.message = '';
            state.status = 'idle';
        },
    },
    extraReducers: (builder) => {
        builder
            // Create Project
            .addCase(createProject.pending, (state) => { state.isLoading = true; })
            .addCase(createProject.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.isLoading = false;
                state.isSuccess = true;
                state.projects.push(action.payload); // Add the new project to the list
                toast.success('Project created successfully!');
            })
            .addCase(createProject.rejected, (state, action) => {
                state.status = 'failed';
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload;
                toast.error(action.payload);
            })

            // Get Projects (List)
            .addCase(getProjects.pending, (state) => { state.isLoading = true; state.status = 'loading'; })
            .addCase(getProjects.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.isLoading = false;
                state.isSuccess = true;
                state.projects = action.payload; // Replace the list with the fetched data
            })
            .addCase(getProjects.rejected, (state, action) => {
                state.status = 'failed';
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload;
                toast.error(action.payload);
            })

            // Get Single Project
            .addCase(getProjectById.pending, (state) => { state.isLoading = true; state.status = 'loading'; })
            .addCase(getProjectById.fulfilled, (state, action) => {
                const project = action.payload || {};
                state.status = 'succeeded';
                state.isLoading = false;
                state.isSuccess = true;
                state.project = {
                    ...project,
                    dbHost: project.dbHost || project.db_host || '',
                };
            })
            .addCase(getProjectById.rejected, (state, action) => {
                state.status = 'failed';
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload;
                toast.error(action.payload);
            })

            // Delete Project
            .addCase(deleteProject.pending, (state) => { state.isLoading = true; state.status = 'loading'; })
            .addCase(deleteProject.fulfilled, (state, action) => {
                state.status = 'succeeded';
                state.isLoading = false;
                state.isSuccess = true;
                // Remove the deleted project from the state array without re-fetching
                state.projects = state.projects.filter(
                    (project) => project.id !== action.payload.id
                );
                toast.success('Project deleted successfully.');
            })
            .addCase(deleteProject.rejected, (state, action) => {
                state.status = 'failed';
                state.isLoading = false;
                state.isError = true;
                state.message = action.payload;
                toast.error(action.payload);
            })
            // Update Project
            .addCase(updateProject.pending, (state) => { state.status = 'loading'; })
            .addCase(updateProject.fulfilled, (state, action) => {
                state.status = 'succeeded';
                toast.success('Project updated!');
                // // Find the project in the array and update it
                // const index = state.projects.findIndex(p => p.id === action.payload.id);
                // if (index !== -1) {
                //     // We only get back the ID, so we can mark it as updated
                //     // A more robust solution might return the full updated project from the API
                //     toast.success('Project updated successfully!');
                // }
            })
            .addCase(updateProject.rejected, (state, action) => {
                state.status = 'failed';
                state.message = action.payload;
                toast.error(action.payload);
            });
    },
});

export const { reset } = projectSlice.actions;
export default projectSlice.reducer;