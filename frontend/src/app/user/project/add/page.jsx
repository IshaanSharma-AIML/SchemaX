'use client';

import { useState } from 'react';
import {
    FaFolder,
    FaDatabase,
    FaUserShield,
    FaKey,
    FaInfoCircle,
    FaRobot,
    FaImage,
    FaArrowRight,
    FaCheckCircle,
    FaServer
} from 'react-icons/fa';
import * as Yup from 'yup';
import { useSelector, useDispatch } from 'react-redux';
import { useRouter } from 'next/navigation';
import { createProject, reset } from '@/lib/store/users-panel/projects/projectSlice';
import WebsiteButton from '@/components/WebsiteButton';

// Dummy data for bot avatars
const avatars = [
    '/bot-avatar-1.png',
    '/bot-avatar-2.png'
];

// Validation schema for the project form
const projectSchema = Yup.object().shape({
    projectName: Yup.string().required('Project Name is required'),
    dbHost: Yup.string().required('Database Host is required'),
    dbUser: Yup.string().required('Database User is required'),
    dbPassword: Yup.string().required('Database Password is required'),
    databaseName: Yup.string().required('Database Name is required'),
    projectInfo: Yup.string(),
    dbInfo: Yup.string(),
    botName: Yup.string(),
});

// Stable sub-components with theme classes
const FormInput = ({ icon, id, label, value, onChange, type = 'text', required = false, placeholder, error }) => (
    <div className="space-y-2">
        <label htmlFor={id} className="text-sm font-medium flex items-center text-gray-700 dark:text-gray-300">
            {icon}
            <span className="ml-2">{label}{required && <span className="text-sky-500 dark:text-sky-400 ml-1">*</span>}</span>
        </label>
        <div className="relative">
            <input
                type={type}
                id={id}
                value={value}
                onChange={onChange}
                className={`w-full px-4 py-3 rounded-lg border outline-none transition
                           bg-gray-50 text-gray-900
                           dark:bg-gray-800 dark:text-white
                           ${error ? 'border-red-500' : 'border-gray-300 dark:border-gray-700'}
                           focus:border-sky-500 focus:ring-2 focus:ring-sky-500/50
                           dark:focus:border-sky-400 dark:focus:ring-sky-400/50`}
                placeholder={placeholder}
            />
        </div>
        {error && <p className="text-red-500 text-xs mt-1">{error}</p>}
    </div>
);

const FormTextArea = ({ icon, id, label, value, onChange, placeholder, rows = 4, error }) => (
    <div className="space-y-2">
        <label htmlFor={id} className="text-sm font-medium flex items-center text-gray-700 dark:text-gray-300">
            {icon}
            <span className="ml-2">{label}</span>
        </label>
        <textarea
            id={id}
            value={value}
            onChange={onChange}
            rows={rows}
            className={`w-full px-4 py-3 rounded-lg border outline-none transition
                       bg-gray-50 text-gray-900
                       dark:bg-gray-800 dark:text-white
                       ${error ? 'border-red-500' : 'border-gray-300 dark:border-gray-700'}
                       focus:border-sky-500 focus:ring-2 focus:ring-sky-500/50
                       dark:focus:border-sky-400 dark:focus:ring-sky-400/50`}
            placeholder={placeholder}
        />
        {error && <p className="text-red-500 text-xs mt-1">{error}</p>}
    </div>
);


const AddProjectPage = () => {
    // State for all form fields
    const [databaseName, setDatabaseName] = useState('');
    const [projectName, setProjectName] = useState('');
    const [projectInfo, setProjectInfo] = useState('');
    const [dbHost, setDbHost] = useState('');
    const [dbUser, setDbUser] = useState('');
    const [dbPassword, setDbPassword] = useState('');
    const [dbPort, setDbPort] = useState('3306');
    const [dbInfo, setDbInfo] = useState('');
    const [botName, setBotName] = useState('');
    const [selectedAvatar, setSelectedAvatar] = useState(avatars[0]);
    const [errors, setErrors] = useState({});

    // Setup Redux hooks
    const dispatch = useDispatch();
    const router = useRouter();

    const { isLoading, isSuccess, isError } = useSelector((state) => state.projects);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const projectData = {
            projectName, projectInfo, dbHost, dbUser, dbPassword, dbPort, databaseName, dbInfo, botName, botAvatar: selectedAvatar
        };
        try {
            await projectSchema.validate(projectData, { abortEarly: false });
            setErrors({});
            await dispatch(createProject(projectData));
            if (isSuccess) {
                router.push('/user/dashboard');
            }
            if (isSuccess || isError) {
                dispatch(reset());
            }
        } catch (validationErrors) {
            const newErrors = {};
            validationErrors.inner.forEach(error => {
                newErrors[error.path] = error.message;
            });
            setErrors(newErrors);
        }
    };

    return (
        <div className="h-full w-full flex flex-col">
            {/* The layout file now handles the main page header */}
            <main className="flex-1 overflow-y-auto">
                <div className="w-full max-w-4xl mx-auto p-4 lg:p-6">
                    <form onSubmit={handleSubmit} className="p-6 lg:p-8 rounded-xl border space-y-6 lg:space-y-8
                                                         bg-white border-gray-200
                                                         dark:bg-gray-900 dark:border-gray-800">

                        {/* --- Section 1: Project Details --- */}
                        <section>
                            <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                         text-gray-900 dark:text-white
                                         border-gray-200 dark:border-gray-700">Project Details</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <FormInput
                                    icon={<FaFolder className="text-gray-500 dark:text-gray-400" />}
                                    id="projectName"
                                    label="Project Name"
                                    value={projectName}
                                    onChange={(e) => setProjectName(e.target.value)}
                                    required
                                    placeholder="e.g., E-commerce Analytics"
                                    error={errors.projectName}
                                />
                                <div className="md:col-span-2">
                                    <FormTextArea
                                        icon={<FaInfoCircle className="text-gray-500 dark:text-gray-400" />}
                                        id="projectInfo"
                                        label="Project Information (Optional)"
                                        value={projectInfo}
                                        onChange={(e) => setProjectInfo(e.target.value)}
                                        placeholder="Briefly describe what this project is for."
                                        rows={3}
                                        error={errors.projectInfo}
                                    />
                                </div>
                            </div>
                        </section>

                        {/* --- Section 2: Database Credentials --- */}
                        <section>
                            <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                         text-gray-900 dark:text-white
                                         border-gray-200 dark:border-gray-700">Database Credentials</h2>
                            <p className="text-xs -mt-4 mb-4 text-gray-600 dark:text-gray-500">
                                Provide your AWS RDS MySQL connection details. Use the endpoint (hostname), admin user, and database you created in RDS.
                            </p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <FormInput
                                    icon={<FaDatabase className="text-gray-500 dark:text-gray-400" />}
                                    id="dbHost"
                                    label="Database Host"
                                    value={dbHost}
                                    onChange={(e) => setDbHost(e.target.value)}
                                    required
                                    placeholder="e.g., my-db.cluster-123456789012.us-east-1.rds.amazonaws.com"
                                    error={errors.dbHost}
                                />
                                <FormInput
                                    icon={<FaUserShield className="text-gray-500 dark:text-gray-400" />}
                                    id="dbUser"
                                    label="Database User"
                                    value={dbUser}
                                    onChange={(e) => setDbUser(e.target.value)}
                                    required
                                    placeholder="e.g., rds_admin"
                                    error={errors.dbUser}
                                />
                                <FormInput
                                    icon={<FaKey className="text-gray-500 dark:text-gray-400" />}
                                    id="dbPassword"
                                    label="Database Password"
                                    type="password"
                                    value={dbPassword}
                                    onChange={(e) => setDbPassword(e.target.value)}
                                    required
                                    placeholder="••••••••"
                                    error={errors.dbPassword}
                                />
                                <FormInput
                                    icon={<FaServer className="text-gray-500 dark:text-gray-400" />}
                                    id="databaseName"
                                    label="Database Name"
                                    value={databaseName}
                                    onChange={(e) => setDatabaseName(e.target.value)}
                                    required
                                    placeholder="e.g., chatbot"
                                    error={errors.databaseName}
                                />
                                <FormInput
                                    icon={<FaDatabase className="text-gray-500 dark:text-gray-400" />}
                                    id="dbPort"
                                    label="Database Port"
                                    value={dbPort}
                                    onChange={(e) => setDbPort(e.target.value)}
                                    placeholder="Default 3306 for AWS RDS MySQL"
                                    error={errors.dbPort}
                                />
                            </div>
                        </section>

                        {/* --- Section 3: AI Agent Configuration --- */}
                        <section>
                            <h2 className="text-xl font-semibold border-b pb-3 mb-6
                                         text-gray-900 dark:text-white
                                         border-gray-200 dark:border-gray-700">AI Agent Configuration</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <FormInput
                                    icon={<FaRobot className="text-gray-500 dark:text-gray-400" />}
                                    id="botName"
                                    label="Give a name to your business bot (Optional)"
                                    value={botName}
                                    onChange={(e) => setBotName(e.target.value)}
                                    placeholder="e.g., SalesBot"
                                />
                                <div className="md:col-span-2">
                                    <FormTextArea
                                        icon={<FaInfoCircle className="text-gray-500 dark:text-gray-400" />}
                                        id="dbInfo"
                                        label="Database Information (For Technical Structure Explanation, Optional)"
                                        value={dbInfo}
                                        onChange={(e) => setDbInfo(e.target.value)}
                                        placeholder="e.g., 'The `users` table contains customer data...'"
                                    />
                                </div>
                                <div className="md:col-span-2">
                                    <label className="text-sm font-medium flex items-center mb-2 text-gray-700 dark:text-gray-300">
                                        <FaImage className="text-gray-500 dark:text-gray-400" />
                                        <span className="ml-2">Choose Avatar (Optional)</span>
                                    </label>
                                    <div className="flex items-center gap-4 mt-2">
                                        {avatars.map((avatarUrl) => (
                                            <div key={avatarUrl} className="relative cursor-pointer" onClick={() => setSelectedAvatar(avatarUrl)}>
                                                <img
                                                    src={avatarUrl}
                                                    alt="Bot Avatar"
                                                    className={`w-16 h-16 rounded-full object-cover border-2 transition ${selectedAvatar === avatarUrl 
                                                        ? 'border-sky-500 dark:border-sky-400' 
                                                        : 'border-gray-200 hover:border-gray-400 dark:border-gray-700 dark:hover:border-gray-500'}`}
                                                />
                                                {selectedAvatar === avatarUrl && (
                                                    <div className="absolute top-0 right-0 w-5 h-5 rounded-full flex items-center justify-center border-2
                                                                  bg-sky-500 dark:bg-sky-400
                                                                  border-white dark:border-gray-900">
                                                        <FaCheckCircle className="text-white text-xs" />
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </section>

                        {/* --- Form Submission --- */}
                        <div className="flex justify-end pt-6 border-t border-gray-200 dark:border-gray-700">
                            <WebsiteButton
                                type="submit"
                                disabled={isLoading}
                            >
                                {isLoading ? 'Creating...' : 'Create Project'}
                                {!isLoading && <FaArrowRight className="ml-2" />}
                            </WebsiteButton>
                        </div>
                    </form>
                </div>
            </main>
        </div>
    );
};

export default AddProjectPage;