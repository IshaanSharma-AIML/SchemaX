'use client';

import { useEffect, useState, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useRouter } from 'next/navigation';
import { getProjects, deleteProject } from '@/lib/store/users-panel/projects/projectSlice';
import { FaPlus, FaFolderPlus, FaProjectDiagram, FaExclamationTriangle, FaEllipsisV, FaPencilAlt, FaTrash, FaEye, FaDatabase, FaComments, FaChartLine } from 'react-icons/fa';
import DeleteConfirmationModal from '@/components/users/modals/DeleteConfirmationModal';

const LoadingSkeleton = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
        {[...Array(3)].map((_, i) => (
            <div key={i} className="rounded-xl p-6 h-36 
                                  bg-gray-100
                                  dark:bg-gray-800">
                <div className="h-4 rounded w-3/4 mb-3
                                bg-gray-200
                                dark:bg-gray-700"></div>
                <div className="h-3 rounded w-1/2
                                bg-gray-200
                                dark:bg-gray-700"></div>
            </div>
        ))}
    </div>
);

const ErrorState = ({ message }) => (
    <div className="rounded-xl p-8 border border-dashed text-center flex flex-col items-center justify-center h-96
                   bg-red-50 border-red-300
                   dark:bg-gray-900 dark:border-red-500/50">
        <div className="w-20 h-20 rounded-full flex items-center justify-center mb-6 border-2
                       bg-red-100 border-red-300
                       dark:bg-red-500/10 dark:border-red-500/50">
            <FaExclamationTriangle className="text-4xl text-red-500 dark:text-red-400" />
        </div>
        <h3 className="text-xl font-semibold mb-2 text-red-800 dark:text-white">Something Went Wrong</h3>
        <p className="mb-6 max-w-sm text-red-700 dark:text-gray-400">
            {message || "We couldn't load your projects. Please try again later."}
        </p>
    </div>
);


const DashboardPage = () => {
    const dispatch = useDispatch();
    const router = useRouter();
    const [openMenuId, setOpenMenuId] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [projectToDelete, setProjectToDelete] = useState(null);
    const menuRef = useRef(null);
    
    const { projects = [], status, message } = useSelector((state) => state.projects);
    const deleteStatus = useSelector((state) => state.projects.status);

    useEffect(() => {
        if (status === 'idle') {
            dispatch(getProjects());
        }
    }, [status, dispatch]);

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

    const handleMenuToggle = (e, projectId) => {
        e.stopPropagation();
        setOpenMenuId(prevId => (prevId === projectId ? null : projectId));
    };

    const handleEdit = (e, projectId) => {
        e.stopPropagation();
        router.push(`/user/project/${projectId}/edit`);
    };

    const handleDeleteClick = (e, project) => {
        e.stopPropagation();
        setProjectToDelete(project);
        setIsModalOpen(true);
        setOpenMenuId(null);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setProjectToDelete(null);
    };

    const handleConfirmDelete = () => {
        if (projectToDelete) {
            dispatch(deleteProject(projectToDelete.id));
        }
        handleCloseModal();
    };

    const renderContent = () => {
        if (status === 'loading' || status === 'idle') {
            return <LoadingSkeleton />;
        }

        if (status === 'failed') {
            return <ErrorState message={message} />;
        }

        if (status === 'succeeded' && projects && projects.length > 0) {
            return (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {projects.map((project) => (
                        <div
                            key={project.id}
                            className="relative bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-all duration-200 group"
                        >
                            <div className="p-6">
                                <div className="flex items-start justify-between mb-4">
                                    <div className="flex items-center">
                                        <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg mr-3">
                                            <FaProjectDiagram className="text-blue-600 dark:text-blue-400 text-lg" />
                                        </div>
                                        <div className="flex-1">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">{project.name}</h3>
                                            <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-2">
                                                {project.project_info || "No description provided."}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => handleMenuToggle(e, project.id)}
                                        className="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                                        title="Options"
                                    >
                                        <FaEllipsisV />
                                    </button>
                                </div>
                                
                                {openMenuId === project.id && (
                                    <div ref={menuRef} className="absolute top-16 right-4 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20">
                                        <ul className="py-2">
                                            <li>
                                                <button 
                                                    onClick={() => router.push(`/user/${project.id}/chat`)} 
                                                    className="w-full text-left px-4 py-2 text-sm flex items-center text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
                                                >
                                                    <FaEye className="mr-3" /> View Project
                                                </button>
                                            </li>
                                            <li>
                                                <button 
                                                    onClick={(e) => handleEdit(e, project.id)} 
                                                    className="w-full text-left px-4 py-2 text-sm flex items-center text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
                                                >
                                                    <FaPencilAlt className="mr-3" /> Edit Project
                                                </button>
                                            </li>
                                            <li>
                                                <button 
                                                    onClick={(e) => handleDeleteClick(e, project)} 
                                                    className="w-full text-left px-4 py-2 text-sm flex items-center text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-gray-700"
                                                >
                                                    <FaTrash className="mr-3" /> Delete Project
                                                </button>
                                            </li>
                                        </ul>
                                    </div>
                                )}
                                
                                <button
                                    onClick={() => router.push(`/user/${project.id}/chat`)}
                                    className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
                                >
                                    Open Project
                                </button>
                            </div>
                        </div>
                    ))}
                    <button
                        onClick={() => router.push('/user/project/add')}
                        className="bg-white dark:bg-gray-800 rounded-xl border-2 border-dashed border-gray-300 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-400 transition-colors flex flex-col items-center justify-center p-8 group"
                        title="Create a new project"
                    >
                        <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg mb-3 group-hover:bg-blue-100 dark:group-hover:bg-blue-900/20 transition-colors">
                            <FaPlus className="text-2xl text-gray-600 dark:text-gray-400 group-hover:text-blue-600 dark:group-hover:text-blue-400" />
                        </div>
                        <span className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400">New Project</span>
                        <span className="text-sm text-gray-500 dark:text-gray-400 mt-1">Create a new AI database project</span>
                    </button>
                </div>
            );
        }

        return (
            <div className="rounded-xl p-8 border border-dashed text-center flex flex-col items-center justify-center h-96
                           bg-gray-50 border-gray-300
                           dark:bg-gray-900 dark:border-gray-700">
                <div className="w-20 h-20 rounded-full flex items-center justify-center mb-6 border-2
                               bg-gray-100 border-gray-300
                               dark:bg-gray-800 dark:border-gray-700">
                    <FaFolderPlus className="text-sky-500 dark:text-sky-400 text-4xl" />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">No Projects Found</h3>
                <p className="mb-6 max-w-sm text-gray-500 dark:text-gray-400">
                    Get started by creating your first project.
                </p>
                <button
                    onClick={() => router.push('/user/project/add')}
                    className="bg-sky-500 cursor-pointer hover:bg-sky-600 text-white font-medium py-3 px-6 rounded-lg flex items-center transition transform hover:scale-105"
                >
                    <FaPlus className="mr-2" />
                    Create New Project
                </button>
            </div>
        );
    };

    return (
        <div className="w-full h-full flex flex-col p-6">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Dashboard</h1>
                <p className="text-lg text-gray-600 dark:text-gray-400">Manage your AI database projects and conversations</p>
            </div>
            
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Projects</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">{projects.length}</p>
                        </div>
                        <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                            <FaProjectDiagram className="text-blue-600 dark:text-blue-400 text-xl" />
                        </div>
                    </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Databases</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">{projects.length}</p>
                        </div>
                        <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-lg">
                            <FaDatabase className="text-green-600 dark:text-green-400 text-xl" />
                        </div>
                    </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Conversations</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">0</p>
                        </div>
                        <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                            <FaComments className="text-purple-600 dark:text-purple-400 text-xl" />
                        </div>
                    </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">Queries</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">0</p>
                        </div>
                        <div className="p-3 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                            <FaChartLine className="text-orange-600 dark:text-orange-400 text-xl" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Projects Section */}
            <div className="flex-1">
                <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-2">Your Projects</h2>
                    <p className="text-gray-600 dark:text-gray-400">Create and manage your AI database projects</p>
                </div>
                
                <DeleteConfirmationModal
                    isOpen={isModalOpen}
                    onClose={handleCloseModal}
                    onConfirm={handleConfirmDelete}
                    title="Delete Project"
                    message={`Are you sure you want to permanently delete the project "${projectToDelete?.name}"? This action cannot be undone.`}
                    isLoading={deleteStatus === 'loading'}
                />
                {renderContent()}
            </div>
        </div>
    );
}

export default DashboardPage;