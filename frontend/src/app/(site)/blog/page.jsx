'use client';

import { FaCalendar, FaUser, FaArrowRight, FaTag, FaSearch, FaFilter } from 'react-icons/fa';
import GradientButton from '@/components/GradientButton';

export default function BlogPage() {
    const blogPosts = [
        {
            id: 1,
            title: "Getting Started with AI Database Queries",
            excerpt: "Learn how to use natural language to query your database without writing SQL code.",
            author: "Sarah Johnson",
            date: "2024-01-15",
            category: "Tutorial",
            readTime: "5 min read",
            image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=250&fit=crop"
        },
        {
            id: 2,
            title: "Top 10 Database Analytics Use Cases",
            excerpt: "Discover the most common ways businesses use AI to analyze their database insights.",
            author: "Mike Chen",
            date: "2024-01-12",
            category: "Analytics",
            readTime: "7 min read",
            image: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=250&fit=crop"
        },
        {
            id: 3,
            title: "Building Interactive Dashboards with AI",
            excerpt: "Create stunning data visualizations using natural language commands and AI assistance.",
            author: "Emily Rodriguez",
            date: "2024-01-10",
            category: "Dashboard",
            readTime: "6 min read",
            image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=250&fit=crop"
        },
        {
            id: 4,
            title: "Database Security Best Practices",
            excerpt: "Essential security measures when connecting AI tools to your database infrastructure.",
            author: "David Kim",
            date: "2024-01-08",
            category: "Security",
            readTime: "8 min read",
            image: "https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=400&h=250&fit=crop"
        }
    ];

    const categories = ["All", "Tutorial", "Analytics", "Dashboard", "Security", "Case Study"];

    return (
        <div className="h-full bg-white dark:bg-gray-900 overflow-hidden">
            <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-y-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                        SchemaX Blog
                    </h1>
                    <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                        Learn how to get the most out of SchemaX AI-powered database analytics and stay updated with the latest trends.
                    </p>
                </div>

                {/* Search and Filter */}
                <div className="flex flex-col md:flex-row gap-4 mb-8">
                    <div className="flex-1 relative">
                        <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search articles..."
                            className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                    </div>
                    <div className="flex gap-2">
                        {categories.map((category) => (
                            <button
                                key={category}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                    category === "All"
                                        ? "bg-blue-600 text-white"
                                        : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                                }`}
                            >
                                {category}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Featured Post */}
                <div className="mb-12">
                    <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
                        <div className="flex items-center gap-2 mb-4">
                            <FaTag className="text-sm" />
                            <span className="text-sm font-medium">Featured</span>
                        </div>
                        <h2 className="text-3xl font-bold mb-4">
                            The Future of Database Analytics: AI-Powered Insights
                        </h2>
                        <p className="text-lg mb-6 opacity-90">
                            Discover how artificial intelligence is revolutionizing the way we interact with databases and extract meaningful insights from our data.
                        </p>
                        <div className="flex items-center gap-4 mb-6">
                            <div className="flex items-center gap-2">
                                <FaUser className="text-sm" />
                                <span className="text-sm">Alex Thompson</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <FaCalendar className="text-sm" />
                                <span className="text-sm">January 20, 2024</span>
                            </div>
                            <span className="text-sm">12 min read</span>
                        </div>
                        <GradientButton href="#" className="bg-white text-blue-600 hover:bg-gray-100">
                            <span className="flex items-center gap-2">
                                Read Article
                                <FaArrowRight className="w-4 h-4" />
                            </span>
                        </GradientButton>
                    </div>
                </div>

                {/* Blog Posts Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {blogPosts.map((post) => (
                        <article key={post.id} className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow border border-gray-200 dark:border-gray-700">
                            <div className="h-48 bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                                <div className="text-white text-4xl font-bold">
                                    {post.title.charAt(0)}
                                </div>
                            </div>
                            <div className="p-6">
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs font-medium rounded-full">
                                        {post.category}
                                    </span>
                                    <span className="text-gray-500 dark:text-gray-400 text-sm">{post.readTime}</span>
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 line-clamp-2">
                                    {post.title}
                                </h3>
                                <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                                    {post.excerpt}
                                </p>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                                        <FaUser className="w-3 h-3" />
                                        <span>{post.author}</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                                        <FaCalendar className="w-3 h-3" />
                                        <span>{post.date}</span>
                                    </div>
                                </div>
                            </div>
                        </article>
                    ))}
                </div>

                {/* Newsletter Signup */}
                <div className="mt-16 bg-gray-50 dark:bg-gray-800 rounded-2xl p-8 text-center">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                        Stay Updated
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300 mb-6">
                        Get the latest articles and insights delivered to your inbox.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
                        <input
                            type="email"
                            placeholder="Enter your email"
                            className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                        <GradientButton href="#" className="px-6 py-3">
                            Subscribe
                        </GradientButton>
                    </div>
                </div>
            </div>
        </div>
    );
}
