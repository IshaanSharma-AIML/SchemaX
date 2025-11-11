'use client';

import React, { useState, useEffect, useRef } from "react";
import { 
    FaBell, 
    FaChevronDown, 
    FaCog, 
    FaSignOutAlt, 
    FaMoon, 
    FaSun, 
    FaUser
} from "react-icons/fa";
import { useDispatch, useSelector } from "react-redux";
import { logout } from "@/lib/store/users-panel/auth/authSlice";
import Link from "next/link";
import { useTheme } from '@/contexts/ThemeContext';
import { useRouter, usePathname, useParams } from 'next/navigation';

const Header = () => {
    const { theme, toggleTheme } = useTheme();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isClient, setIsClient] = useState(false);
    const menuRef = useRef(null);
    const dispatch = useDispatch();
    const router = useRouter();
    const params = useParams();
    const projectId = params?.projectId;

    // Get user data from Redux store
    const { user } = useSelector((state) => state.auth);

    // Ensure we're on the client side to prevent hydration issues
    useEffect(() => {
        setIsClient(true);
    }, []);

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [menuRef]);

    const handleLogout = () => {
        dispatch(logout());
        router.push('/login');
    };


    const getUserInitials = () => {
        if (!isClient) {
            return 'U'; // Default for server-side rendering
        }
        if (user?.name) {
            return user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
        }
        return 'U';
    };

    return (
        <header className="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl border-b border-gray-200/50 dark:border-gray-700/50 shadow-sm">
            <div className="w-full px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-24">
                    {/* Left side: Logo + Navigation */}
                    <div className="flex items-center gap-8">
                        {/* Logo */}
                        <Link href="/user/dashboard" className="group">
                            <div className="w-48 h-48 flex items-center justify-center 
                                group-hover:scale-105 transition-all duration-300">
                                <img 
                                    src={theme === 'light' ? "/light.png" : "/dark.png"} 
                                    alt="SchemaX Logo" 
                                    className="w-48 h-48 object-contain"
                                />
                            </div>
                        </Link>

                    </div>

                    {/* Right side: Actions */}
                    <div className="flex items-center gap-3">
                        {/* Theme Toggle */}
                        <button
                            onClick={toggleTheme}
                            className="p-2.5 rounded-lg transition-all duration-200
                                text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/50
                                group"
                            aria-label="Toggle theme"
                        >
                            {theme === 'light' ? (
                                <FaMoon className="text-sm group-hover:scale-110 transition-transform" />
                            ) : (
                                <FaSun className="text-sm group-hover:scale-110 transition-transform" />
                            )}
                        </button>

                        {/* Notifications */}
                        <div className="relative group">
                            <button
                                type="button"
                                aria-label="Notifications"
                                className="p-2.5 rounded-lg transition-all duration-200
                                    text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                    dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/50
                                    group"
                            >
                                <FaBell className="text-sm group-hover:scale-110 transition-transform" />
                                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                            </button>
                        </div>

                        {/* User Menu */}
                        <div className="relative" ref={menuRef}>
                            <button
                                type="button"
                                onClick={() => setIsMenuOpen(!isMenuOpen)}
                                className="flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200
                                    bg-gray-50 hover:bg-gray-100 dark:bg-gray-800 dark:hover:bg-gray-700
                                    group"
                            >
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center shadow-sm">
                                    <span className="text-white text-xs font-bold">{getUserInitials()}</span>
                                </div>
                                <div className="hidden sm:block text-left">
                                    <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                        {isClient ? (user?.name || 'Loading...') : 'Loading...'}
                                    </div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">
                                        {isClient ? (user?.email || 'Loading...') : 'Loading...'}
                                    </div>
                                </div>
                                <FaChevronDown className={`text-xs transition-transform duration-200 ${
                                    isMenuOpen ? 'rotate-180' : ''
                                } text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-300`} />
                            </button>

                            {/* User Dropdown Menu */}
                            {isMenuOpen && (
                                <div className="absolute right-0 mt-2 w-64 rounded-xl shadow-xl z-50 
                                    bg-white border border-gray-200 dark:bg-gray-800 dark:border-gray-700
                                    overflow-hidden">
                                    {/* User Info */}
                                    <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                                                <span className="text-white text-sm font-bold">{getUserInitials()}</span>
                                            </div>
                                            <div>
                                                <div className="text-sm font-medium text-gray-900 dark:text-white">
                                                    {isClient ? (user?.name || 'Loading...') : 'Loading...'}
                                                </div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400">
                                                    {isClient ? (user?.email || 'Loading...') : 'Loading...'}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Menu Items */}
                                    <div className="py-2">
                                        <Link
                                            href="/user/profile"
                                            onClick={() => setIsMenuOpen(false)}
                                            className="flex items-center gap-3 px-4 py-2 text-sm transition-colors
                                                text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700"
                                        >
                                            <FaUser className="text-xs" />
                                            Profile
                                        </Link>
                                        <Link
                                            href={projectId ? `/user/project/${projectId}/edit` : "/user/settings"}
                                            onClick={() => setIsMenuOpen(false)}
                                            className="flex items-center gap-3 px-4 py-2 text-sm transition-colors
                                                text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700"
                                        >
                                            <FaCog className="text-xs" />
                                            {projectId ? 'Project Settings' : 'Settings'}
                                        </Link>
                                        <hr className="my-2 border-gray-200 dark:border-gray-700" />
                                        <button
                                            onClick={handleLogout}
                                            className="w-full flex items-center gap-3 px-4 py-2 text-sm transition-colors
                                                text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
                                        >
                                            <FaSignOutAlt className="text-xs" />
                                            Sign Out
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>

                    </div>
                </div>

            </div>
        </header>
    );
};

export default Header;