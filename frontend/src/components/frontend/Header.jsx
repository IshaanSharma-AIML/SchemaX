"use client";
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { 
    FaBrain, 
    FaMoon, 
    FaSun, 
    FaBars, 
    FaTimes,
    FaRocket,
    FaChartLine,
    FaShieldAlt,
    FaUsers,
    FaCog
} from 'react-icons/fa';
import GradientButton from '@/components/GradientButton';
import { useTheme } from '@/contexts/ThemeContext';

const Header = () => {
    const { theme, toggleTheme } = useTheme();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isScrolled, setIsScrolled] = useState(false);

    // Handle scroll effect
    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navigationItems = [];

    return (
        <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 w-full ${
            isScrolled 
                ? 'bg-white dark:bg-gray-900/80 shadow-lg border-b border-gray-200/50 dark:border-gray-800/30' 
                : 'bg-white dark:bg-gray-900/80'
        }`}>
            <div className="w-full px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16 lg:h-20">
                    {/* Logo */}
                    <Link href="/" className="group">
                        <div className="w-40 h-40 lg:w-48 lg:h-48 flex items-center justify-center 
                            group-hover:scale-105 transition-all duration-300">
                            <img 
                                src={theme === 'light' ? "/light.png" : "/dark.png"} 
                                alt="SchemaX Logo" 
                                className="w-40 h-40 lg:w-48 lg:h-48 object-contain"
                            />
                        </div>
                    </Link>

                    {/* Desktop Navigation */}
                    <nav className="hidden lg:flex items-center space-x-1">
                        {navigationItems.map((item) => (
                            <a
                                key={item.name}
                                href={item.href}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                                    text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                    dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30
                                    group"
                            >
                                <item.icon className="text-xs opacity-60 group-hover:opacity-100 transition-opacity" />
                                {item.name}
                            </a>
                        ))}
                    </nav>

                    {/* Right Side Actions */}
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

                        {/* Desktop Auth Buttons */}
                        <div className="hidden sm:flex items-center gap-3">
                            {/* <Link 
                                href="/pricing" 
                                className="px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200
                                    text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                    dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30"
                            >
                                Pricing
                            </Link> */}
                            <Link 
                                href="/login" 
                                className="px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200
                                    text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                    dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30"
                            >
                                Sign In
                            </Link>
                            <GradientButton 
                                href="/register" 
                                className="px-6 py-2.5 text-sm font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
                            >
                                Get Started Free
                            </GradientButton>
                        </div>

                        {/* Mobile Menu Button */}
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="lg:hidden p-2.5 rounded-lg transition-all duration-200
                                text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/50"
                            aria-label="Toggle menu"
                        >
                            {isMenuOpen ? (
                                <FaTimes className="text-lg" />
                            ) : (
                                <FaBars className="text-lg" />
                            )}
                        </button>
                    </div>
                </div>

                {/* Mobile Navigation Menu */}
                {isMenuOpen && (
                    <div className="lg:hidden border-t border-gray-200/50 dark:border-gray-800/30 bg-white/95 dark:bg-gray-900/80 backdrop-blur-xl">
                        <div className="px-4 py-6 space-y-4">
                            {navigationItems.map((item) => (
                                <a
                                    key={item.name}
                                    href={item.href}
                                    onClick={() => setIsMenuOpen(false)}
                                    className="flex items-center gap-3 px-4 py-3 rounded-lg text-base font-medium transition-all duration-200
                                        text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                        dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30"
                                >
                                    <item.icon className="text-sm" />
                                    {item.name}
                                </a>
                            ))}
                            
                            {/* Mobile Auth Buttons */}
                            <div className="pt-4 border-t border-gray-200/50 dark:border-gray-800/30 space-y-3">
                                {/* <Link 
                                    href="/pricing" 
                                    onClick={() => setIsMenuOpen(false)}
                                    className="block w-full px-4 py-3 text-center text-base font-medium rounded-lg transition-all duration-200
                                        text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                        dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30"
                                >
                                    Pricing
                                </Link> */}
                                <Link 
                                    href="/login" 
                                    onClick={() => setIsMenuOpen(false)}
                                    className="block w-full px-4 py-3 text-center text-base font-medium rounded-lg transition-all duration-200
                                        text-gray-600 hover:text-gray-900 hover:bg-gray-100/50
                                        dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-800/30"
                                >
                                    Sign In
                                </Link>
                                <GradientButton 
                                    href="/register" 
                                    onClick={() => setIsMenuOpen(false)}
                                    className="block w-full px-4 py-3 text-center text-base font-semibold rounded-lg shadow-lg"
                                >
                                    Get Started Free
                                </GradientButton>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </header>
    );
};

export default Header;
