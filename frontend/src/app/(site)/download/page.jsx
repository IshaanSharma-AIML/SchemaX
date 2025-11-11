'use client';

import { FaDownload, FaDesktop, FaMobile, FaCloud, FaShieldAlt, FaAward, FaCheckCircle, FaArrowRight } from 'react-icons/fa';
import GradientButton from '@/components/GradientButton';

export default function DownloadPage() {
    const downloadOptions = [
        {
            id: 'desktop',
            title: 'Desktop App',
            description: 'Secure local database connections with full offline capabilities',
            icon: <FaDesktop className="text-4xl text-blue-500" />,
            features: [
                'Local database connections',
                'Offline mode support',
                'Enhanced security',
                'Full feature access',
                'No data transmission'
            ],
            downloadText: 'Download for Windows',
            downloadLink: '#',
            size: '45 MB',
            version: 'v2.1.0'
        },
        {
            id: 'mobile',
            title: 'Mobile App',
            description: 'Access your database insights on the go with our mobile application',
            icon: <FaMobile className="text-4xl text-green-500" />,
            features: [
                'iOS and Android support',
                'Push notifications',
                'Offline viewing',
                'Touch-optimized interface',
                'Secure authentication'
            ],
            downloadText: 'Download for Mobile',
            downloadLink: '#',
            size: '25 MB',
            version: 'v1.8.2'
        },
        {
            id: 'web',
            title: 'Web App',
            description: 'Access AskYourDatabase directly from your browser - no installation required',
            icon: <FaCloud className="text-4xl text-purple-500" />,
            features: [
                'No installation needed',
                'Cross-platform compatibility',
                'Real-time collaboration',
                'Automatic updates',
                'Cloud synchronization'
            ],
            downloadText: 'Launch Web App',
            downloadLink: '/register',
            size: 'Instant',
            version: 'Latest'
        }
    ];

    const systemRequirements = {
        desktop: {
            windows: {
                os: 'Windows 10 or later',
                ram: '4 GB RAM minimum',
                storage: '100 MB free space',
                processor: 'Intel Core i3 or equivalent'
            },
            mac: {
                os: 'macOS 10.15 or later',
                ram: '4 GB RAM minimum',
                storage: '100 MB free space',
                processor: 'Intel Core i3 or Apple M1'
            },
            linux: {
                os: 'Ubuntu 18.04 or later',
                ram: '4 GB RAM minimum',
                storage: '100 MB free space',
                processor: 'Intel Core i3 or equivalent'
            }
        },
        mobile: {
            ios: {
                os: 'iOS 13.0 or later',
                ram: '2 GB RAM minimum',
                storage: '50 MB free space',
                device: 'iPhone 8 or later'
            },
            android: {
                os: 'Android 8.0 or later',
                ram: '2 GB RAM minimum',
                storage: '50 MB free space',
                device: 'Any Android device'
            }
        }
    };

    return (
        <div className="h-full bg-white dark:bg-gray-900 overflow-hidden">
            <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-y-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                        Download SchemaX
                    </h1>
                    <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                        Choose the version that works best for you. All versions include the same powerful AI features to help you analyze your database.
                    </p>
                </div>

                {/* Download Options */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
                    {downloadOptions.map((option) => (
                        <div key={option.id} className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
                            <div className="text-center mb-6">
                                {option.icon}
                                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mt-4 mb-2">
                                    {option.title}
                                </h3>
                                <p className="text-gray-600 dark:text-gray-300 mb-4">
                                    {option.description}
                                </p>
                                <div className="flex justify-center items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                                    <span>Size: {option.size}</span>
                                    <span>•</span>
                                    <span>Version: {option.version}</span>
                                </div>
                            </div>

                            <div className="space-y-3 mb-8">
                                {option.features.map((feature, index) => (
                                    <div key={index} className="flex items-center gap-3">
                                        <FaCheckCircle className="text-green-500 flex-shrink-0" />
                                        <span className="text-gray-700 dark:text-gray-300 text-sm">{feature}</span>
                                    </div>
                                ))}
                            </div>

                            <GradientButton 
                                href={option.downloadLink} 
                                className="w-full py-3"
                            >
                                <span className="flex items-center justify-center gap-2">
                                    <FaDownload className="w-4 h-4" />
                                    {option.downloadText}
                                </span>
                            </GradientButton>
                        </div>
                    ))}
                </div>

                {/* System Requirements */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                    {/* Desktop Requirements */}
                    <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
                        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
                            <FaDesktop className="text-blue-500" />
                            Desktop Requirements
                        </h3>
                        
                        <div className="space-y-6">
                            {Object.entries(systemRequirements.desktop).map(([platform, requirements]) => (
                                <div key={platform} className="border-l-4 border-blue-500 pl-4">
                                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2 capitalize">
                                        {platform}
                                    </h4>
                                    <div className="space-y-1 text-sm text-gray-600 dark:text-gray-300">
                                        <p><strong>OS:</strong> {requirements.os}</p>
                                        <p><strong>RAM:</strong> {requirements.ram}</p>
                                        <p><strong>Storage:</strong> {requirements.storage}</p>
                                        <p><strong>Processor:</strong> {requirements.processor}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Mobile Requirements */}
                    <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
                        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
                            <FaMobile className="text-green-500" />
                            Mobile Requirements
                        </h3>
                        
                        <div className="space-y-6">
                            {Object.entries(systemRequirements.mobile).map(([platform, requirements]) => (
                                <div key={platform} className="border-l-4 border-green-500 pl-4">
                                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2 capitalize">
                                        {platform}
                                    </h4>
                                    <div className="space-y-1 text-sm text-gray-600 dark:text-gray-300">
                                        <p><strong>OS:</strong> {requirements.os}</p>
                                        <p><strong>RAM:</strong> {requirements.ram}</p>
                                        <p><strong>Storage:</strong> {requirements.storage}</p>
                                        <p><strong>Device:</strong> {requirements.device}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Security & Trust */}
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white text-center">
                    <h3 className="text-2xl font-bold mb-4">Secure & Trusted</h3>
                    <p className="text-lg mb-6 opacity-90">
                        All downloads are verified and secure. Your data privacy is our top priority.
                    </p>
                </div>

                {/* FAQ */}
                <div className="mt-12">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
                        Frequently Asked Questions
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                                Is the desktop app free?
                            </h4>
                            <p className="text-gray-600 dark:text-gray-300 text-sm">
                                Yes, the desktop app is free to download and use with basic features. Premium features are available with a subscription.
                            </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                                Can I use multiple versions?
                            </h4>
                            <p className="text-gray-600 dark:text-gray-300 text-sm">
                                Absolutely! You can use the web app, desktop app, and mobile app simultaneously with the same account.
                            </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                                How often are updates released?
                            </h4>
                            <p className="text-gray-600 dark:text-gray-300 text-sm">
                                We release updates monthly with new features, bug fixes, and security improvements.
                            </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                                Is my data secure?
                            </h4>
                            <p className="text-gray-600 dark:text-gray-300 text-sm">
                                Yes, all versions use enterprise-grade encryption and follow strict security protocols to protect your data.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
