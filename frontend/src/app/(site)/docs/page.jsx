'use client';

import { FaBook, FaCode, FaRocket, FaQuestionCircle, FaArrowRight, FaSearch, FaFilter, FaExternalLinkAlt } from 'react-icons/fa';
import GradientButton from '@/components/GradientButton';

export default function DocsPage() {
    const docCategories = [
        {
            id: 'getting-started',
            title: 'Getting Started',
            description: 'Quick setup and first steps with AskYourDatabase',
            icon: <FaRocket className="text-2xl text-blue-500" />,
            articles: [
                { title: 'Installation Guide', description: 'Set up AskYourDatabase in minutes', readTime: '5 min' },
                { title: 'First Database Connection', description: 'Connect your first database', readTime: '3 min' },
                { title: 'Your First Query', description: 'Ask your first question', readTime: '2 min' }
            ]
        },
        {
            id: 'api',
            title: 'API Reference',
            description: 'Complete API documentation and examples',
            icon: <FaCode className="text-2xl text-green-500" />,
            articles: [
                { title: 'Authentication', description: 'API keys and authentication methods', readTime: '4 min' },
                { title: 'Query Endpoints', description: 'Send queries and get responses', readTime: '6 min' },
                { title: 'Webhook Integration', description: 'Set up real-time notifications', readTime: '5 min' }
            ]
        },
        {
            id: 'guides',
            title: 'Guides & Tutorials',
            description: 'Step-by-step tutorials and best practices',
            icon: <FaBook className="text-2xl text-purple-500" />,
            articles: [
                { title: 'Building Dashboards', description: 'Create interactive data dashboards', readTime: '8 min' },
                { title: 'Advanced Queries', description: 'Complex query patterns and tips', readTime: '10 min' },
                { title: 'Data Visualization', description: 'Generate charts and graphs', readTime: '7 min' }
            ]
        },
        {
            id: 'troubleshooting',
            title: 'Troubleshooting',
            description: 'Common issues and solutions',
            icon: <FaQuestionCircle className="text-2xl text-orange-500" />,
            articles: [
                { title: 'Connection Issues', description: 'Fix database connection problems', readTime: '5 min' },
                { title: 'Query Errors', description: 'Debug and fix query issues', readTime: '6 min' },
                { title: 'Performance Tips', description: 'Optimize your queries', readTime: '4 min' }
            ]
        }
    ];

    const popularArticles = [
        { title: 'Quick Start Guide', category: 'Getting Started', readTime: '3 min', views: '12.5k' },
        { title: 'Database Connection Setup', category: 'Getting Started', readTime: '5 min', views: '8.2k' },
        { title: 'API Authentication', category: 'API Reference', readTime: '4 min', views: '6.8k' },
        { title: 'Building Your First Dashboard', category: 'Guides', readTime: '8 min', views: '5.4k' },
        { title: 'Troubleshooting Connection Issues', category: 'Troubleshooting', readTime: '5 min', views: '4.9k' }
    ];

    return (
        <div className="h-full bg-white dark:bg-gray-900 overflow-hidden">
            <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-y-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                        SchemaX Documentation
                    </h1>
                    <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                        Everything you need to get started with SchemaX and build powerful AI-powered database applications.
                    </p>
                </div>

                {/* Search */}
                <div className="max-w-2xl mx-auto mb-12">
                    <div className="relative">
                        <FaSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search documentation..."
                            className="w-full pl-12 pr-4 py-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
                        />
                    </div>
                </div>

                {/* Popular Articles */}
                <div className="mb-12">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Popular Articles</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {popularArticles.map((article, index) => (
                            <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                                <div className="flex items-start justify-between mb-3">
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs font-medium rounded-full">
                                        {article.category}
                                    </span>
                                    <span className="text-xs text-gray-500 dark:text-gray-400">{article.views} views</span>
                                </div>
                                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{article.title}</h3>
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-gray-500 dark:text-gray-400">{article.readTime} read</span>
                                    <FaExternalLinkAlt className="text-gray-400 text-sm" />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Documentation Categories */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {docCategories.map((category) => (
                        <div key={category.id} className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
                            <div className="flex items-center gap-4 mb-6">
                                {category.icon}
                                <div>
                                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">{category.title}</h3>
                                    <p className="text-gray-600 dark:text-gray-300 text-sm">{category.description}</p>
                                </div>
                            </div>
                            
                            <div className="space-y-4">
                                {category.articles.map((article, index) => (
                                    <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors cursor-pointer">
                                        <div className="flex-1">
                                            <h4 className="font-medium text-gray-900 dark:text-white mb-1">{article.title}</h4>
                                            <p className="text-sm text-gray-600 dark:text-gray-300">{article.description}</p>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className="text-xs text-gray-500 dark:text-gray-400">{article.readTime}</span>
                                            <FaArrowRight className="text-gray-400 text-sm" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Quick Links */}
                <div className="mt-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
                    <div className="text-center">
                        <h3 className="text-2xl font-bold mb-4">Need Help?</h3>
                        <p className="text-lg mb-6 opacity-90">
                            Can't find what you're looking for? Our support team is here to help.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <GradientButton href="/contact" className="bg-white text-blue-600 hover:bg-gray-100">
                                <span className="flex items-center gap-2">
                                    Contact Support
                                    <FaArrowRight className="w-4 h-4" />
                                </span>
                            </GradientButton>
                            <GradientButton href="/register" className="bg-transparent border-2 border-white text-white hover:bg-white hover:text-blue-600">
                                <span className="flex items-center gap-2">
                                    Get Started
                                    <FaRocket className="w-4 h-4" />
                                </span>
                            </GradientButton>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
