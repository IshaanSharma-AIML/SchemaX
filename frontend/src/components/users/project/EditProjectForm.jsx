'use client';

import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useParams, useRouter } from 'next/navigation';
import { getProjectById, updateProject, reset } from '@/lib/store/users-panel/projects/projectSlice';
import * as Yup from 'yup';
import { FaFolder, FaDatabase, FaUserShield, FaKey, FaInfoCircle, FaRobot, FaImage, FaArrowRight, FaCheckCircle, FaServer } from 'react-icons/fa';

// Reusable Input Component with theme classes
const FormInput = ({ icon, id, label, value, onChange, type = 'text', placeholder, error }) => (
    <div className="space-y-2">
        <label htmlFor={id} className="text-sm font-medium flex items-center text-gray-700 dark:text-gray-300">
            {icon}
            <span className="ml-2">{label}</span>
        </label>
        <div className="relative">
            <input
                type={type}
                id={id}
                name={id}
                value={value}
                onChange={onChange}
                className={`w-full px-4 py-3 rounded-lg border outline-none transition
                           bg-gray-50 text-gray-900 dark:bg-gray-800 dark:text-white
                           ${error ? 'border-red-500' : 'border-gray-300 dark:border-gray-700'}
                           focus:border-sky-500 focus:ring-2 focus:ring-sky-500/50
                           dark:focus:border-sky-400 dark:focus:ring-sky-400/50`}
                placeholder={placeholder}
            />
        </div>
        {error && <p className="text-red-500 text-xs mt-1">{error}</p>}
    </div>
);

// Reusable Textarea Component with theme classes
const FormTextArea = ({ icon, id, label, value, onChange, placeholder, rows = 4, error }) => (
    <div className="space-y-2">
        <label htmlFor={id} className="text-sm font-medium flex items-center text-gray-700 dark:text-gray-300">
            {icon}
            <span className="ml-2">{label}</span>
        </label>
        <textarea
            id={id}
            name={id}
            value={value}
            onChange={onChange}
            rows={rows}
            className={`w-full px-4 py-3 rounded-lg border outline-none transition
                       bg-gray-50 text-gray-900 dark:bg-gray-800 dark:text-white
                       ${error ? 'border-red-500' : 'border-gray-300 dark:border-gray-700'}
                       focus:border-sky-500 focus:ring-2 focus:ring-sky-500/50
                       dark:focus:border-sky-400 dark:focus:ring-sky-400/50`}
            placeholder={placeholder}
        />
        {error && <p className="text-red-500 text-xs mt-1">{error}</p>}
    </div>
);

// Dummy avatar data
const avatars = [
    '/bot-avatar-1.png',
    '/bot-avatar-2.png',
];

// Validation schema for editing
const editProjectSchema = Yup.object().shape({
    projectName: Yup.string().required('Project Name is required'),
    // Database fields are optional in edit mode - user can leave blank to keep unchanged
    dbHost: Yup.string(),
    dbUser: Yup.string(),
    dbPassword: Yup.string(),
    dbPort: Yup.string(),
    databaseName: Yup.string(),
});

const EditProjectForm = () => {
    const [formData, setFormData] = useState({
        projectName: '', projectInfo: '', dbHost: '', dbUser: '', dbPassword: '', dbPort: '3306', databaseName: '', dbInfo: '', botName: '',
    });
    const [selectedAvatar, setSelectedAvatar] = useState('');
    const [errors, setErrors] = useState({});

    const dispatch = useDispatch();
    const router = useRouter();
    const params = useParams();
    const { projectId } = params;

    const { project, status, isLoading, isSuccess, isError } = useSelector((state) => state.projects);

    useEffect(() => {
        if (projectId) {
            dispatch(getProjectById(projectId));
        }
    }, [projectId, dispatch]);

    useEffect(() => {
        if (project && project.id === projectId) {
            setFormData({
                projectName: project.name || '',
                projectInfo: project.project_info || '',
                dbInfo: project.db_info || '',
                botName: project.bot_name || '',
                dbHost: '', // Leave blank - user can enter new host
                dbUser: '', // Leave blank - user can enter new username
                dbPassword: '', // Leave blank - user can enter new password
                dbPort: project.db_port || '3306', // Show current port
                databaseName: '', // Leave blank - user can enter new database name
            });
            setSelectedAvatar(project.bot_avatar || '');
        }
    }, [project, projectId]);

    // Form data is handled securely - no sensitive data logging

    const handleInputChange = (e) => {
        const { id, value } = e.target;
        setFormData(prev => ({ ...prev, [id]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await editProjectSchema.validate(formData, { abortEarly: false });
            setErrors({});
            const dataToSubmit = Object.fromEntries(
                Object.entries(formData).filter(([key, value]) => value !== '')
            );
            dataToSubmit.botAvatar = selectedAvatar;
            
            await dispatch(updateProject({ projectId, projectData: dataToSubmit }));
            if (isSuccess) {
                router.push('/user/dashboard');
            }
            if (isSuccess || isError) {
                dispatch(reset());
            }

        } catch (validationErrors) {
            const newErrors = {};
            validationErrors.inner.forEach(error => { newErrors[error.path] = error.message; });
            setErrors(newErrors);
        }
    };

    return (
        <div className="h-full w-full flex flex-col">
            <main className="flex-1 overflow-y-auto">
                <div className="w-full max-w-4xl mx-auto p-4 lg:p-6">
                    {status === 'loading' && !formData.projectName ? (
                        <div className="p-6 text-center text-gray-700 dark:text-white">Loading project details...</div>
                    ) : (
                        <form onSubmit={handleSubmit} className="p-6 lg:p-8 rounded-xl border space-y-6 lg:space-y-8
                                                             bg-white border-gray-200
                                                             dark:bg-gray-900 dark:border-gray-800">
                            <section>
                                <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                             text-gray-900 dark:text-white
                                             border-gray-200 dark:border-gray-700">Project Details</h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput icon={<FaFolder className="text-gray-500 dark:text-gray-400"/>} id="projectName" label="Project Name" value={formData.projectName} onChange={handleInputChange} error={errors.projectName} />
                                    <div className="md:col-span-2">
                                        <FormTextArea icon={<FaInfoCircle className="text-gray-500 dark:text-gray-400"/>} id="projectInfo" label="Project Information (Optional)" value={formData.projectInfo} onChange={handleInputChange} rows={3} error={errors.projectInfo} />
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                             text-gray-900 dark:text-white
                                             border-gray-200 dark:border-gray-700">Database Credentials</h2>
                                <p className="text-xs -mt-4 mb-4 text-gray-600 dark:text-gray-500">Leave fields blank to keep them unchanged.</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput icon={<FaDatabase className="text-gray-500 dark:text-gray-400"/>} id="dbHost" label="Database Host" value={formData.dbHost} onChange={handleInputChange} error={errors.dbHost} placeholder="Enter new host (e.g., my-db.cluster-123456789012.us-east-1.rds.amazonaws.com)" />
                                    <FormInput icon={<FaUserShield className="text-gray-500 dark:text-gray-400"/>} id="dbUser" label="Database User" value={formData.dbUser} onChange={handleInputChange} error={errors.dbUser} placeholder="Enter new username (e.g., rds_admin)" />
                                    <FormInput icon={<FaKey className="text-gray-500 dark:text-gray-400"/>} id="dbPassword" type="password" label="Database Password" value={formData.dbPassword} onChange={handleInputChange} error={errors.dbPassword} placeholder="Enter new password" />
                                    <FormInput icon={<FaServer className="text-gray-500 dark:text-gray-400"/>} id="databaseName" label="Database Name" value={formData.databaseName} onChange={handleInputChange} error={errors.databaseName} placeholder="Enter new database name (e.g., chatbot)" />
                                    <FormInput icon={<FaDatabase className="text-gray-500 dark:text-gray-400"/>} id="dbPort" label="Database Port" value={formData.dbPort} onChange={handleInputChange} error={errors.dbPort} placeholder="Default 3306 for AWS RDS MySQL" />
                                </div>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                             text-gray-900 dark:text-white
                                             border-gray-200 dark:border-gray-700">AI Agent Configuration</h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput icon={<FaRobot className="text-gray-500 dark:text-gray-400"/>} id="botName" label="AI Agent Name (Optional)" value={formData.botName} onChange={handleInputChange} error={errors.botName} />
                                    <div className="md:col-span-2">
                                        <FormTextArea icon={<FaInfoCircle className="text-gray-500 dark:text-gray-400"/>} id="dbInfo" label="Database Information (Optional)" value={formData.dbInfo} onChange={handleInputChange} error={errors.dbInfo} />
                                    </div>
                                    <div className="md:col-span-2">
                                        <label className="text-sm font-medium flex items-center mb-2 text-gray-700 dark:text-gray-300"><FaImage className="text-gray-500 dark:text-gray-400" /><span className="ml-2">Choose Avatar</span></label>
                                        <div className="flex items-center gap-4 mt-2">
                                            {avatars.map((avatarUrl) => (
                                                <div key={avatarUrl} className="relative cursor-pointer" onClick={() => setSelectedAvatar(avatarUrl)}>
                                                    <img src={avatarUrl} alt="Bot Avatar" className={`w-16 h-16 rounded-full object-cover border-2 transition ${selectedAvatar === avatarUrl ? 'border-sky-500 dark:border-sky-400' : 'border-gray-200 hover:border-gray-400 dark:border-gray-700 dark:hover:border-gray-500'}`} />
                                                    {selectedAvatar === avatarUrl && (<div className="absolute top-0 right-0 w-5 h-5 rounded-full flex items-center justify-center border-2 bg-sky-500 dark:bg-sky-400 border-white dark:border-gray-900"><FaCheckCircle className="text-white text-xs" /></div>)}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </section>

                            <div className="flex justify-end pt-6 border-t border-gray-200 dark:border-gray-700">
                                <button type="submit" className="bg-sky-500 hover:bg-sky-600 text-white font-medium py-3 px-6 rounded-lg flex items-center transition transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed" disabled={isLoading}>
                                    {isLoading ? 'Updating...' : 'Save Changes'}
                                    {!isLoading && <FaArrowRight className="ml-2" />}
                                </button>
                            </div>
                        </form>
                    )}
                </div>
            </main>
        </div>
    );
};

export default EditProjectForm;