'use client';

import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { 
    getVisualizations, 
    getVisualization, 
    toggleFavoriteVisualization, 
    deleteVisualization 
} from '@/lib/store/users-panel/chat/chatSlice';
import { FaHeart, FaTrash, FaDownload, FaEye, FaChartBar, FaCalendarAlt, FaDatabase } from 'react-icons/fa';
import { format } from 'date-fns';

const VisualizationManager = ({ projectId, conversationId, onVisualizationSelect }) => {
    const dispatch = useDispatch();
    const { visualizations, visualizationsStatus } = useSelector(state => state.chat);
    const [selectedVisualization, setSelectedVisualization] = useState(null);
    const [filter, setFilter] = useState('all'); // 'all', 'favorites', 'recent'
    const [sortBy, setSortBy] = useState('created_at'); // 'created_at', 'title', 'chart_type'

    useEffect(() => {
        if (projectId) {
            dispatch(getVisualizations({ 
                projectId, 
                conversationId,
                limit: 100 
            }));
        }
    }, [dispatch, projectId, conversationId]);

    const handleVisualizationClick = async (visualization) => {
        try {
            await dispatch(getVisualization(visualization.id)).unwrap();
            setSelectedVisualization(visualization);
            if (onVisualizationSelect) {
                onVisualizationSelect(visualization);
            }
        } catch (error) {
            console.error('Failed to load visualization:', error);
        }
    };

    const handleToggleFavorite = async (visualizationId, e) => {
        e.stopPropagation();
        try {
            await dispatch(toggleFavoriteVisualization(visualizationId)).unwrap();
        } catch (error) {
            console.error('Failed to toggle favorite:', error);
        }
    };

    const handleDelete = async (visualizationId, e) => {
        e.stopPropagation();
        if (window.confirm('Are you sure you want to delete this visualization?')) {
            try {
                await dispatch(deleteVisualization(visualizationId)).unwrap();
            } catch (error) {
                console.error('Failed to delete visualization:', error);
            }
        }
    };

    const handleDownload = (visualization, e) => {
        e.stopPropagation();
        try {
            const link = document.createElement('a');
            link.href = `data:image/png;base64,${visualization.chart_data}`;
            link.download = `${visualization.title.replace(/[^a-zA-Z0-9]/g, '_')}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Failed to download visualization:', error);
        }
    };

    const filteredVisualizations = visualizations.filter(viz => {
        if (filter === 'favorites') return viz.is_favorite;
        if (filter === 'recent') {
            const oneWeekAgo = new Date();
            oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
            return new Date(viz.created_at) > oneWeekAgo;
        }
        return true;
    });

    const sortedVisualizations = [...filteredVisualizations].sort((a, b) => {
        switch (sortBy) {
            case 'title':
                return a.title.localeCompare(b.title);
            case 'chart_type':
                return a.chart_type.localeCompare(b.chart_type);
            case 'created_at':
            default:
                return new Date(b.created_at) - new Date(a.created_at);
        }
    });

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

    if (visualizationsStatus === 'loading') {
        return (
            <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-gray-600">Loading visualizations...</span>
            </div>
        );
    }

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            {/* Header */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <FaChartBar className="mr-2" />
                        Stored Visualizations
                    </h3>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                        {sortedVisualizations.length} visualization{sortedVisualizations.length !== 1 ? 's' : ''}
                    </span>
                </div>
            </div>

            {/* Filters and Sort */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700">
                <div className="flex flex-wrap gap-4 items-center">
                    <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Filter:</label>
                        <select
                            value={filter}
                            onChange={(e) => setFilter(e.target.value)}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                            <option value="all">All</option>
                            <option value="favorites">Favorites</option>
                            <option value="recent">Recent (7 days)</option>
                        </select>
                    </div>
                    <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Sort by:</label>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value)}
                            className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                        >
                            <option value="created_at">Date Created</option>
                            <option value="title">Title</option>
                            <option value="chart_type">Chart Type</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Visualizations List */}
            <div className="max-h-96 overflow-y-auto">
                {sortedVisualizations.length === 0 ? (
                    <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                        <FaChartBar className="mx-auto text-4xl mb-4 opacity-50" />
                        <p>No visualizations found</p>
                        <p className="text-sm">Create some visualizations in your conversations to see them here</p>
                    </div>
                ) : (
                    <div className="divide-y divide-gray-200 dark:divide-gray-700">
                        {sortedVisualizations.map((visualization) => (
                            <div
                                key={visualization.id}
                                className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                                onClick={() => handleVisualizationClick(visualization)}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center space-x-2 mb-2">
                                            <span className="text-lg">
                                                {getChartTypeInfo(visualization.chart_type).icon}
                                            </span>
                                            <h4 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                                                {visualization.title}
                                            </h4>
                                            <span className={`px-2 py-1 text-xs rounded-full bg-${getChartTypeInfo(visualization.chart_type).color}-100 text-${getChartTypeInfo(visualization.chart_type).color}-800 dark:bg-${getChartTypeInfo(visualization.chart_type).color}-900 dark:text-${getChartTypeInfo(visualization.chart_type).color}-200`}>
                                                {getChartTypeInfo(visualization.chart_type).label}
                                            </span>
                                            {visualization.is_favorite && (
                                                <FaHeart className="text-red-500 text-xs" />
                                            )}
                                        </div>
                                        
                                        <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400 mb-2">
                                            <span className="flex items-center">
                                                <FaDatabase className="mr-1" />
                                                {visualization.project_name}
                                            </span>
                                            <span className="flex items-center">
                                                <FaCalendarAlt className="mr-1" />
                                                {format(new Date(visualization.created_at), 'MMM d, yyyy')}
                                            </span>
                                        </div>
                                        
                                        {visualization.query_used && (
                                            <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                                                Query: {visualization.query_used}
                                            </p>
                                        )}
                                        
                                        {/* Mini preview of the visualization */}
                                        {visualization.chart_data && (
                                            <div className="mt-2 flex justify-center">
                                                <img 
                                                    src={`data:image/png;base64,${visualization.chart_data}`}
                                                    alt={visualization.title}
                                                    className="w-16 h-12 object-cover rounded border border-gray-200 dark:border-gray-600 opacity-75 hover:opacity-100 transition-opacity"
                                                />
                                            </div>
                                        )}
                                    </div>
                                    
                                    <div className="flex items-center space-x-2 ml-4">
                                        <button
                                            onClick={(e) => handleToggleFavorite(visualization.id, e)}
                                            className={`p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors ${
                                                visualization.is_favorite ? 'text-red-500' : 'text-gray-400'
                                            }`}
                                            title={visualization.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
                                        >
                                            <FaHeart className="text-xs" />
                                        </button>
                                        
                                        <button
                                            onClick={(e) => handleDownload(visualization, e)}
                                            className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-gray-400"
                                            title="Download visualization"
                                        >
                                            <FaDownload className="text-xs" />
                                        </button>
                                        
                                        <button
                                            onClick={(e) => handleDelete(visualization.id, e)}
                                            className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-gray-400 hover:text-red-500"
                                            title="Delete visualization"
                                        >
                                            <FaTrash className="text-xs" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default VisualizationManager;
