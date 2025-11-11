'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDispatch, useSelector } from 'react-redux';
import { registerUser, reset } from '@/lib/store/users-panel/auth/authSlice';
import { FaUser, FaEnvelope, FaLock, FaEye, FaEyeSlash, FaArrowRight, FaGift } from 'react-icons/fa';
import { useTheme } from '@/contexts/ThemeContext';

export default function RegisterPage() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
    });
    const [showPassword, setShowPassword] = useState(false);
    const router = useRouter();
    const dispatch = useDispatch();
    const { theme } = useTheme();
    
    const { isLoading, isError, message, isSuccess } = useSelector((state) => state.auth);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!formData.name || !formData.email || !formData.password) {
            alert('Please fill in all fields');
            return;
        }
        
        if (formData.password.length < 8) {
            alert('Password must be at least 8 characters long');
            return;
        }

        // Reset any previous errors
        dispatch(reset());
        
        try {
            // Try real API call first
            const result = await dispatch(registerUser(formData));
            
            if (registerUser.fulfilled.match(result)) {
                // Registration successful, navigate to dashboard
                router.push('/user/dashboard');
            }
        } catch (error) {
            // Fallback for demo: if API fails, use demo mode
            console.log('API registration failed, using demo mode');
            
            // Set demo auth data
            const demoUser = { id: 1, email: formData.email, name: formData.name };
            const demoToken = 'demo-token-' + Date.now();
            
            // Store in localStorage and cookies
            localStorage.setItem('user', JSON.stringify(demoUser));
            localStorage.setItem('token', demoToken);
            document.cookie = `authToken=${demoToken}; path=/; max-age=86400`;
            
            // Navigate to dashboard
            router.push('/user/dashboard');
        }
    };

    return (
        <div className="h-full w-full bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Modern Glass Card */}
                <div className="relative bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border border-white/20 dark:border-gray-700/50">
                    {/* Decorative Elements */}
                    <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-purple-400 to-pink-500 rounded-full opacity-20 blur-xl"></div>
                    <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full opacity-20 blur-xl"></div>
                    
                    <div className="relative">
                                <div className="text-center mb-8">
                                    {/* Modern Logo */}
                                    <div className="w-48 h-48 mx-auto mb-4 flex items-center justify-center">
                                        <img 
                                            src={theme === 'light' ? "/light.png" : "/dark.png"} 
                                            alt="SchemaX Logo" 
                                            className="w-48 h-48 object-contain"
                                        />
                                    </div>
                                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Get started</h1>
                                    <p className="text-gray-600 dark:text-gray-400">Create your SchemaX account</p>
                                    
                                    {/* Error Message */}
                                    {isError && message && (
                                        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                                            <p className="text-red-600 dark:text-red-400 text-sm">{message}</p>
                                        </div>
                                    )}
                                </div>


                        {/* Free Trial Banner */}
                        <div className="mb-4 p-3 bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-200/50 dark:border-green-700/50 rounded-2xl">
                            <div className="flex items-center justify-center gap-2 text-sm text-green-700 dark:text-green-300">
                                <FaGift className="w-4 h-4" />
                                <span className="font-semibold">14-day free trial • No credit card required</span>
                            </div>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <label htmlFor="name" className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                                    Full name
                                </label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <FaUser className="text-gray-400" />
                                    </div>
                                    <input
                                        type="text"
                                        id="name"
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-4 rounded-2xl border-2 border-gray-200 dark:border-gray-600 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 outline-none transition-all duration-200"
                                        placeholder="Enter your full name"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="email" className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                                    Email address
                                </label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <FaEnvelope className="text-gray-400" />
                                    </div>
                                    <input
                                        type="email"
                                        id="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-4 rounded-2xl border-2 border-gray-200 dark:border-gray-600 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 outline-none transition-all duration-200"
                                        placeholder="Enter your email"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="password" className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                                    Password
                                </label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <FaLock className="text-gray-400" />
                                    </div>
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        id="password"
                                        name="password"
                                        value={formData.password}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-12 py-4 rounded-2xl border-2 border-gray-200 dark:border-gray-600 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 outline-none transition-all duration-200"
                                        placeholder="Create a password"
                                        required
                                        minLength={8}
                                    />
                                    <button
                                        type="button"
                                        className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                                        onClick={() => setShowPassword(!showPassword)}
                                    >
                                        {showPassword ? <FaEyeSlash className="w-5 h-5" /> : <FaEye className="w-5 h-5" />}
                                    </button>
                                </div>
                                <p className="text-xs text-gray-500 dark:text-gray-400">Minimum 8 characters</p>
                            </div>

                            {/* Modern Button */}
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full py-4 px-6 bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 disabled:bg-gray-400 text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-3 disabled:cursor-not-allowed disabled:transform-none"
                            >
                                {isLoading ? (
                                    <span>Creating account...</span>
                                ) : (
                                    <>
                                        <span>Create account</span>
                                        <FaArrowRight className="w-4 h-4" />
                                    </>
                                )}
                            </button>
                        </form>

                        <div className="mt-6 text-center">
                            <p className="text-gray-600 dark:text-gray-400">
                                Already have an account?{' '}
                                <a href="/login" className="text-purple-600 dark:text-purple-400 font-semibold hover:text-purple-700 dark:hover:text-purple-300 transition-colors">
                                    Sign in
                                </a>
                            </p>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
}