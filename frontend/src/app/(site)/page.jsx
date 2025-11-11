// /src/app/page.jsx

'use client';

import Link from 'next/link';
import {
    FaBrain, FaDatabase, FaComments, FaChartLine, FaLink,
    FaLanguage, FaUserSecret, FaCheckCircle, FaPlayCircle,
    FaRocket, FaShieldAlt, FaUsers, FaArrowRight, FaAward, FaClock, FaLock
} from 'react-icons/fa';
import GradientButton from '@/components/GradientButton';
import { useTheme } from '@/contexts/ThemeContext';

// --- Hero Section ---
const HeroSection = () => (
    <section className="h-full flex items-center justify-center text-center px-4 w-full relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
            {/* Floating Orbs */}
            <div className="absolute top-20 left-10 w-20 h-20 bg-blue-500/10 rounded-full animate-pulse"></div>
            <div className="absolute top-40 right-20 w-16 h-16 bg-purple-500/10 rounded-full animate-bounce" style={{animationDelay: '1s'}}></div>
            <div className="absolute bottom-40 left-20 w-24 h-24 bg-cyan-500/10 rounded-full animate-pulse" style={{animationDelay: '2s'}}></div>
            <div className="absolute bottom-20 right-10 w-12 h-12 bg-pink-500/10 rounded-full animate-bounce" style={{animationDelay: '0.5s'}}></div>
            
            {/* Gradient Mesh */}
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-gradient-to-r from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '3s'}}></div>
            <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-gradient-to-r from-cyan-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1.5s'}}></div>
        </div>
        
        <div className="relative w-full max-w-none z-10">
            
            {/* Main Headline */}
                    <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold text-gray-900 dark:text-white leading-tight mb-6 animate-fade-in-up">
                        <span className="inline-block animate-fade-in-up" style={{animationDelay: '0.2s'}}>SchemaX</span>
                        <span className="block text-blue-600 dark:text-blue-400 animate-fade-in-up" style={{animationDelay: '0.4s'}}>Your Data, Your Language</span>
            </h1>
            
            {/* Subtitle */}
                    <div className="mb-8 animate-fade-in-up" style={{animationDelay: '0.6s'}}>
                        <h2 className="text-2xl md:text-3xl font-bold text-gray-800 dark:text-gray-200 mb-4">
                            Ask Questions, Get <span className="text-blue-600 dark:text-blue-400 animate-pulse">Instant Insights</span>
                        </h2>
                        <p className="text-lg md:text-xl text-gray-600 dark:text-gray-300 w-full leading-relaxed">
                            Transform your database into a conversational partner. Just ask, and watch your data come alive.
                        </p>
                    </div>

            {/* Description */}
            <p className="text-base md:text-lg text-gray-700 dark:text-gray-300 mb-8 w-full leading-relaxed animate-fade-in-up" style={{animationDelay: '0.8s'}}>
                Experience the future of data interaction. From complex queries to beautiful visualizations, SchemaX makes your database as easy to use as a conversation.
            </p>

            {/* Database Logos */}
            <div className="mb-8 animate-fade-in-up" style={{animationDelay: '1s'}}>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">Works seamlessly with your favorite databases:</p>
                <div className="flex flex-wrap justify-center items-center gap-8 opacity-80">
                    {/* PostgreSQL */}
                    <div className="flex items-center justify-center p-4 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                        <img 
                            src="https://wiki.postgresql.org/images/a/a4/PostgreSQL_logo.3colors.svg" 
                            alt="PostgreSQL" 
                            className="w-16 h-16 object-contain"
                        />
                    </div>
                    
                    {/* MySQL */}
                    <div className="flex items-center justify-center p-4 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                        <img 
                            src="https://dev.mysql.com/common/logos/mysql-logo.svg" 
                            alt="MySQL" 
                            className="w-16 h-16 object-contain"
                        />
                    </div>
                    
                    {/* MongoDB */}
                    <div className="flex items-center justify-center p-4 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                        <img 
                            src="https://webassets.mongodb.com/_com_assets/cms/mongodb_logo1-76twgcu2dm.png" 
                            alt="MongoDB" 
                            className="w-16 h-16 object-contain"
                        />
                    </div>
                </div>
            </div>
        
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row justify-center items-center gap-4 mb-12 animate-fade-in-up" style={{animationDelay: '1.2s'}}>
                <GradientButton href="/register" className="py-4 px-8 text-lg font-semibold group shadow-lg bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 transition-all duration-300 rounded-lg transform hover:scale-105 hover:shadow-2xl animate-pulse-slow">
                    <span className="flex items-center gap-2">
                        Get Started
                        <FaArrowRight className="group-hover:translate-x-1 transition-transform duration-300" />
                    </span>
                </GradientButton>
            </div>
        </div>
    </section>
);

// --- Features Section ---
const FeaturesSection = () => (
    <section id="features" className="py-16 px-4 bg-gray-50 dark:bg-gray-800 w-full relative overflow-hidden">
        {/* Background Animation */}
        <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-10 left-10 w-32 h-32 bg-blue-500/5 rounded-full animate-pulse" style={{animationDelay: '2s'}}></div>
            <div className="absolute bottom-10 right-10 w-40 h-40 bg-purple-500/5 rounded-full animate-pulse" style={{animationDelay: '4s'}}></div>
        </div>
        
        <div className="w-full relative z-10">
            <div className="text-center mb-12 animate-fade-in-up">
                <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                    Business Intelligence Made Simple
                </h2>
                <p className="text-lg text-gray-600 dark:text-gray-300 w-full">
                    No more juggling between developers and BI tools. Just ask, and get insights instantly.
                </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                    {
                        icon: <FaChartLine className="text-4xl text-blue-500" />,
                        title: "Business Intelligence",
                        description: "No more juggling between developers and BI tools. Just ask, and get insights instantly.",
                        features: ["Real-time analytics", "Custom dashboards", "Automated reports"]
                    },
                    {
                        icon: <FaDatabase className="text-4xl text-green-500" />,
                        title: "Data Visualization",
                        description: "Instantly transform complex data into clear, engaging visuals. No coding needed, just insights at a glance.",
                        features: ["Interactive charts", "Custom visualizations", "Export capabilities"]
                    },
                    {
                        icon: <FaRocket className="text-4xl text-purple-500" />,
                        title: "Dashboard Builder",
                        description: "Create real-time, interactive dashboards using natural language. No SQL or complex query language required.",
                        features: ["Natural language queries", "Real-time updates", "Shareable dashboards"]
                    }
                ].map((feature, idx) => (
                    <div key={idx} className="bg-white dark:bg-gray-900 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-center mb-6">
                            {feature.icon}
                        </div>
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 text-center">
                            {feature.title}
                        </h3>
                        <p className="text-gray-600 dark:text-gray-300 mb-6 text-center">
                            {feature.description}
                        </p>
                        <ul className="space-y-2">
                            {feature.features.map((item, i) => (
                                <li key={i} className="flex items-center text-sm text-gray-600 dark:text-gray-300">
                                    <FaCheckCircle className="text-green-500 mr-2 flex-shrink-0" />
                                    {item}
                                </li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </div>
    </section>
);

// --- Database Support Section ---
const DatabaseSupportSection = () => (
    <section className="py-16 px-4 w-full">
        <div className="w-full">
            <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                    Works with your favorite databases
                </h2>
                <p className="text-lg text-gray-600 dark:text-gray-300">
                    Connect to any major database and start chatting with your data immediately
                </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-8 items-center justify-items-center">
                {[
                    { name: "PostgreSQL", color: "text-blue-600" },
                    { name: "MySQL", color: "text-orange-600" },
                    { name: "SQL Server", color: "text-red-600" },
                    { name: "Oracle", color: "text-red-500" },
                    { name: "MongoDB", color: "text-green-600" },
                    { name: "Snowflake", color: "text-blue-500" }
                ].map((db, idx) => (
                    <div key={idx} className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700">
                        <FaDatabase className={`text-3xl ${db.color} mb-2`} />
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{db.name}</span>
                    </div>
                ))}
            </div>
            
            <div className="text-center mt-8">
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Don't have a database? Create one in Neon DB
                </p>
                <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors">
                    Create Database
                </button>
            </div>
        </div>
    </section>
);

// --- Testimonials Section ---
const TestimonialsSection = () => (
    <section className="py-16 px-4 bg-gray-50 dark:bg-gray-800 w-full">
        <div className="w-full">
            <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                    Trusted by 100+ companies worldwide
                </h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {[
                    {
                        quote: "This tool has saved us at least 40 hours of front and back work each week with the CS team and engineering team, but more importantly has taught the CS team a very important skill that's crucial for their growth and success",
                        author: "Vaibhav Namburi",
                        role: "Founder of SmartLead.ai",
                        avatar: "VN"
                    },
                    {
                        quote: "Just a few minutes with AskYourDatabase and I'm already seeing results that surpass what I've experienced with other SQL AI tools! The efficiency and insight are next-level!",
                        author: "Lionel Thomas",
                        role: "Real Estate Pro, Appraiser",
                        avatar: "LT"
                    },
                    {
                        quote: "There's been a lot of innovation with ChatGPT's plugins. One that has caught my eye is AskYourDatabase. It helps to create complex SQL or Python scripts. With the plugin, you connect straight to your data – and can chat with it!",
                        author: "Tom Taulli",
                        role: "Author of Generative AI book",
                        avatar: "TT"
                    }
                ].map((testimonial, idx) => (
                    <div key={idx} className="bg-white dark:bg-gray-900 rounded-xl p-8 shadow-lg border border-gray-200 dark:border-gray-700">
                        <div className="mb-6">
                            <p className="text-gray-700 dark:text-gray-300 italic mb-4">
                                "{testimonial.quote}"
                            </p>
                        </div>
                        <div className="flex items-center">
                            <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center text-white font-bold mr-4">
                                {testimonial.avatar}
                            </div>
                            <div>
                                <div className="font-semibold text-gray-900 dark:text-white">
                                    {testimonial.author}
                                </div>
                                <div className="text-sm text-gray-600 dark:text-gray-400">
                                    {testimonial.role}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    </section>
);

// --- Footer Section ---
const FooterSection = ({ theme }) => (
    <footer className="bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white py-16 px-4 w-full">
        <div className="w-full">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 items-start">
                {/* Company Info */}
                <div className="col-span-1 md:col-span-2">
                    <div className="mb-4">
                        <div className="w-32 h-32 flex items-center justify-center">
                            <img 
                                src={theme === 'light' ? "/light.png" : "/dark.png"} 
                                alt="SchemaX Logo" 
                                className="w-32 h-32 object-contain"
                            />
                        </div>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400 mb-6 w-full">
                        The best AI for databases. No SQL knowledge needed, connect your database and chat with your data using natural language.
                    </p>
                    <div className="flex gap-4">
                        <a href="/register" className="bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200">
                            Get Started Free
                        </a>
                        {/* <a href="/pricing" className="border border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 text-gray-700 dark:text-white px-6 py-2 rounded-lg font-medium transition-colors">
                            View Pricing
                        </a> */}
                    </div>
                </div>
                
                {/* Product */}
                <div>
                    <h4 className="font-semibold mb-4 text-gray-900 dark:text-white">Product</h4>
                    <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                        <li><a href="#features" className="hover:text-gray-900 dark:hover:text-white transition-colors">Features</a></li>
                        {/* <li><a href="/pricing" className="hover:text-gray-900 dark:hover:text-white transition-colors">Pricing</a></li> */}
                        {/* <li><a href="/download" className="hover:text-gray-900 dark:hover:text-white transition-colors">Download</a></li> */}
                        <li><a href="/docs" className="hover:text-gray-900 dark:hover:text-white transition-colors">Documentation</a></li>
                    </ul>
                </div>
                
                {/* Support */}
                <div>
                    <h4 className="font-semibold mb-4 text-gray-900 dark:text-white">Support</h4>
                    <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                        <li><a href="/contact" className="hover:text-gray-900 dark:hover:text-white transition-colors">Contact Us</a></li>
                        <li><a href="/blog" className="hover:text-gray-900 dark:hover:text-white transition-colors">Blog</a></li>
                        {/* <li><a href="#" className="hover:text-gray-900 dark:hover:text-white transition-colors">Help Center</a></li>
                        <li><a href="#" className="hover:text-gray-900 dark:hover:text-white transition-colors">Status</a></li> */}
                    </ul>
                </div>
            </div>
            
            {/* Bottom Bar */}
            <div className="border-t border-gray-200 dark:border-gray-700 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                    © 2025 SchemaX. All rights reserved.
                </p>
                <div className="flex gap-6 mt-4 md:mt-0">
                    <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-sm transition-colors">Privacy Policy</a>
                    <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-sm transition-colors">Terms of Service</a>
                    <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white text-sm transition-colors">Cookie Policy</a>
                </div>
            </div>
        </div>
    </footer>
);


// --- Main Component ---
export default function HomePage() {
    const { theme } = useTheme();
    
    return (
        <div className="antialiased bg-white dark:bg-gray-900 h-full overflow-y-auto">
                <HeroSection />
                <FeaturesSection />
            <FooterSection theme={theme} />
        </div>
    );
}
